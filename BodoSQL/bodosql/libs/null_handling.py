# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Library of BodoSQL functions used for performing operations on potentially null Values
"""
from numba import generated_jit
from numba.extending import register_jitable

import bodo


@generated_jit(nopython=True)
def scalar_nullable_logical_not(val):
    """Helper function that performs a logical not on a nullable boolean scalar value"""
    if val == bodo.none:
        return lambda val: None
    # If the input is optional, the output is optional.
    # We could merge this code path with the default, but
    # if we can avoid optional types we should.
    elif isinstance(val, bodo.optional):

        def impl(val):
            if val is None:
                return None
            else:
                # Call internal bodo function that changes the converts the
                # type of Optional(type) to just type.
                return not bodo.utils.indexing.unoptional(val)

        return impl
    else:
        # Note: We rely on the type checking for not here to catch errors
        return lambda val: not val


@generated_jit(nopython=True)
def scalar_nullable_add(a, b):
    """
    Add operator on scalars with SQL Null handling. This function can take None,
    which should return None (different from Python), an Optional type, which
    should return None if the value is None at runtime or compute the add if
    neither value is none, or a + b.
    """
    # If either input is None, return None
    if a == bodo.none or b == bodo.none:
        return lambda a, b: None
    # If either input is optional, the output is optional.
    # We could merge this code path with the default, but
    # if we can avoid optional types we should.
    elif isinstance(a, bodo.optional) or isinstance(b, bodo.optional):

        def impl(a, b):
            if a is None or b is None:
                return None
            else:
                # Call internal bodo function that changes the converts the
                # type of Optional(type) to just type. If a or b isn't optional
                # this is basically a noop
                return bodo.utils.indexing.unoptional(
                    a
                ) + bodo.utils.indexing.unoptional(b)

        return impl
    else:
        # Note: We rely on the type checking for + here to catch errors
        return lambda a, b: a + b


@register_jitable
def null_if_not_flag(val, flag):
    """
    Helper function to handle cases where nested if else
    statements should be used in BodoSQL.
    """
    return val if flag else None
