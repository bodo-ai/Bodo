# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Library of BodoSQL functions used for performing "AND" and "OR" operations involving potentially null values
"""

from numba import generated_jit

import bodo


@generated_jit(nopython=True)
def mysql_nullchecking_scalar_or(arg0, arg1):
    "Function that replicates the behavior of MYSQL's or function on scalar values"

    # in the null case,
    # TRUE OR NULL = TRUE
    # FALSE OR NULL = NULL
    # NULL OR TRUE = TRUE
    # see https://spark.apache.org/docs/3.0.0-preview/sql-ref-null-semantics.html#logical-operators

    if isinstance(arg0, bodo.optional) or isinstance(arg1, bodo.optional):

        def impl(arg0, arg1):
            if arg0 is None and arg1 is None:
                return None
            elif arg0 is None:
                arg1 = bodo.utils.indexing.unoptional(arg1)
                return True if arg1 else None
            elif arg1 is None:
                arg0 = bodo.utils.indexing.unoptional(arg0)
                return True if arg0 else None

            else:
                # Call internal bodo function that changes the converts the
                # type of Optional(type) to just type. If a or b isn't optional
                # this is basically a noop
                arg0 = bodo.utils.indexing.unoptional(arg0)
                arg1 = bodo.utils.indexing.unoptional(arg1)
                return arg0 or arg1

        return impl
    elif arg0 == bodo.none and arg1 == bodo.none:
        return lambda arg0, arg1: None
    elif arg0 == bodo.none:
        return lambda arg0, arg1: True if arg1 else None
    elif arg1 == bodo.none:
        return lambda arg0, arg1: True if arg0 else None
    else:
        return lambda arg0, arg1: arg0 or arg1


@generated_jit(nopython=True)
def mysql_nullchecking_scalar_and(arg0, arg1):
    "Function that replicates the behavior of MYSQL's and function on scalar values"

    # in the null case,
    # FALSE AND NULL = FALSE
    # TRUE AND NULL = NULL
    # NULL AND FALSE = NULL
    # see https://spark.apache.org/docs/3.0.0-preview/sql-ref-null-semantics.html#logical-operators
    if arg0 == bodo.none:
        return lambda arg0, arg1: None
    elif isinstance(arg0, bodo.optional) or isinstance(arg1, bodo.optional):

        def impl(arg0, arg1):
            if arg0 is None:
                return None
            elif arg1 is None:
                arg0 = bodo.utils.indexing.unoptional(arg0)
                return False if arg0 is False else None

            else:
                # Call internal bodo function that changes the converts the
                # type of Optional(type) to just type. If a or b isn't optional
                # this is basically a noop
                arg0 = bodo.utils.indexing.unoptional(arg0)
                arg1 = bodo.utils.indexing.unoptional(arg1)
                return arg0 and arg1

        return impl
    elif arg1 == bodo.none:
        # arg0 != None
        return lambda arg0, arg1: False if arg0 is False else None
    else:
        return lambda arg0, arg1: arg0 and arg1
