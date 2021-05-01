# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Array of intervals corresponding to IntervalArray of Pandas.
Used for IntervalIndex, which is necessary for Series.value_counts() with 'bins'
argument.
"""

import pandas as pd
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    register_model,
    typeof_impl,
    unbox,
)

import bodo


class IntervalArrayType(types.ArrayCompatible):
    """data type corresponding to IntervalArray of Pandas"""

    def __init__(self, arr_type):
        # array type for left and right arrays, which have the same data type
        # see IntervalArray.from_arrays
        # https://github.com/pandas-dev/pandas/blob/c65ed1c40fce55198fcd67c2bef15ab88645c1fa/pandas/core/arrays/interval.py#L277
        self.arr_type = arr_type
        super(IntervalArrayType, self).__init__(name=f"IntervalArrayType({arr_type})")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return IntervalArrayType(self.arr_type)


@register_model(IntervalArrayType)
class IntervalArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # assuming closed="right", TODO: support closed
        members = [
            ("left", fe_type.arr_type),
            ("right", fe_type.arr_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IntervalArrayType, "left", "_left")
make_attribute_wrapper(IntervalArrayType, "right", "_right")


@typeof_impl.register(pd.arrays.IntervalArray)
def typeof_interval_array(val, c):
    arr_type = bodo.typeof(val._left)
    return IntervalArrayType(arr_type)


@intrinsic
def init_interval_array(typingctx, left, right=None):
    """Create a IntervalArray with provided left and right arrays."""
    assert left == right, "Interval left/right array types should be the same"

    def codegen(context, builder, signature, args):
        left_val, right_val = args
        # create interval_arr struct and store values
        interval_arr = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        interval_arr.left = left_val
        interval_arr.right = right_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], left_val)
        context.nrt.incref(builder, signature.args[1], right_val)

        return interval_arr._getvalue()

    ret_typ = IntervalArrayType(left)
    sig = ret_typ(left, right)
    return sig, codegen


@box(IntervalArrayType)
def box_interval_arr(typ, val, c):
    """
    Box interval array into IntervalArray object of Pandas
    """
    interval_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    # box left
    c.context.nrt.incref(c.builder, typ.arr_type, interval_arr.left)
    left_obj = c.pyapi.from_native_value(typ.arr_type, interval_arr.left, c.env_manager)
    # box right
    c.context.nrt.incref(c.builder, typ.arr_type, interval_arr.right)
    right_obj = c.pyapi.from_native_value(
        typ.arr_type, interval_arr.right, c.env_manager
    )

    # call pd.arrays.IntervalArray.from_arrays(left, right)
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    arrays_mod_obj = c.pyapi.object_getattr_string(pd_class_obj, "arrays")
    interval_array_class_obj = c.pyapi.object_getattr_string(
        arrays_mod_obj, "IntervalArray"
    )
    interval_array_obj = c.pyapi.call_method(
        interval_array_class_obj, "from_arrays", (left_obj, right_obj)
    )

    c.pyapi.decref(left_obj)
    c.pyapi.decref(right_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(arrays_mod_obj)
    c.pyapi.decref(interval_array_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return interval_array_obj


@unbox(IntervalArrayType)
def unbox_interval_arr(typ, val, c):
    """
    Unbox IntervalArray object to native value
    """
    # get left array
    left_obj = c.pyapi.object_getattr_string(val, "_left")
    left = c.pyapi.to_native_value(typ.arr_type, left_obj).value
    c.pyapi.decref(left_obj)

    # get right array
    right_obj = c.pyapi.object_getattr_string(val, "_right")
    right = c.pyapi.to_native_value(typ.arr_type, right_obj).value
    c.pyapi.decref(right_obj)

    # create interval array
    interval_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    interval_arr.left = left
    interval_arr.right = right

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(interval_arr._getvalue(), is_error=is_error)


@overload(len, no_unliteral=True)
def overload_interval_arr_len(A):
    if isinstance(A, IntervalArrayType):
        return lambda A: len(A._left)  # pragma: no cover


@overload_attribute(IntervalArrayType, "shape")
def overload_interval_arr_shape(A):
    return lambda A: (len(A._left),)  # pragma: no cover


@overload_attribute(IntervalArrayType, "ndim")
def overload_interval_arr_ndim(A):
    return lambda A: 1  # pragma: no cover
