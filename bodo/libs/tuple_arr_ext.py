# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Array of tuple values, implemented by reusing array of structs implementation.
"""
import datetime
import decimal
import glob
import operator
import warnings

import llvmlite.llvmpy.core as lc
import numba
import numba.core.typing.typeof
import numpy as np
import pandas as pd
from numba.core import cgutils, types
from numba.core.imputils import (
    RefType,
    impl_ret_borrowed,
    impl_ret_new_ref,
    iternext_impl,
    lower_constant,
)
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)

import bodo
from bodo.hiframes.datetime_date_ext import (
    datetime_date_array_type,
    datetime_date_type,
)
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayPayloadType,
    ArrayItemArrayType,
    _get_array_item_arr_payload,
)
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.struct_arr_ext import (
    StructArrayType,
    box_struct_arr,
    unbox_struct_array,
)
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    is_list_like_index_type,
    is_overload_constant_int,
    is_overload_none,
    is_overload_true,
    parse_dtype,
)


class TupleArrayType(types.ArrayCompatible):
    """Data type for arrays of tuples"""

    def __init__(self, data):
        # data is tuple of Array types
        self.data = data
        super(TupleArrayType, self).__init__(name="TupleArrayType({})".format(data))

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        # TODO(ehsan): support namedtuples
        return types.BaseTuple.from_types(tuple(t.dtype for t in self.data))

    def copy(self):
        return TupleArrayType(self.data)


@register_model(TupleArrayType)
class TupleArrayModel(models.StructModel):
    """Use struct array to store tuple array data"""

    def __init__(self, dmm, fe_type):
        members = [
            ("data", StructArrayType(fe_type.data)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(TupleArrayType, "data", "_data")


@intrinsic
def init_tuple_arr(typingctx, data_typ=None):
    """create a new tuple array from struct array data"""
    assert isinstance(data_typ, StructArrayType)
    out_type = TupleArrayType(data_typ.data)

    def codegen(context, builder, sig, args):
        (data_arr,) = args
        tuple_array = context.make_helper(builder, out_type)
        tuple_array.data = data_arr
        context.nrt.incref(builder, data_typ, data_arr)
        return tuple_array._getvalue()

    return out_type(data_typ), codegen


@unbox(TupleArrayType)
def unbox_tuple_array(typ, val, c):
    """
    Unbox a numpy array with tuple values.
    """
    # reuse struct array implementation
    data_typ = StructArrayType(typ.data)
    struct_native_val = unbox_struct_array(data_typ, val, c, is_tuple_array=True)
    data_arr = struct_native_val.value
    tuple_array = c.context.make_helper(c.builder, typ)
    tuple_array.data = data_arr
    is_error = struct_native_val.is_error
    return NativeValue(tuple_array._getvalue(), is_error=is_error)


@box(TupleArrayType)
def box_tuple_arr(typ, val, c):
    """box tuple array into python objects"""
    # reuse struct array implementation
    data_typ = StructArrayType(typ.data)
    tuple_array = c.context.make_helper(c.builder, typ, val)
    arr = box_struct_arr(data_typ, tuple_array.data, c, is_tuple_array=True)
    # NOTE: no need to decref since box_struct_arr decrefs all data
    return arr
