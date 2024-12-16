"""Array of tuple values, implemented by reusing array of structs implementation."""

import operator

import numba
import numpy as np
from numba.core import types
from numba.extending import (
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs.struct_arr_ext import (
    StructArrayType,
)
from bodo.utils.typing import BodoError, is_list_like_index_type


class TupleArrayType(types.ArrayCompatible):
    """Data type for arrays of tuples"""

    def __init__(self, data):
        # data is tuple of Array types
        self.data = data
        super().__init__(name=f"TupleArrayType({data})")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        # TODO(ehsan): support namedtuples
        return types.BaseTuple.from_types(tuple(t.dtype for t in self.data))

    def copy(self):
        return TupleArrayType(self.data)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


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
def init_tuple_arr(typingctx, data_typ):
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
    Unbox an array with tuple values.
    """
    return bodo.libs.array.unbox_array_using_arrow(typ, val, c)


@box(TupleArrayType)
def box_tuple_arr(typ, val, c):
    """box tuple array into python objects"""
    return bodo.libs.array.box_array_using_arrow(typ, val, c)


@numba.njit
def pre_alloc_tuple_array(n, nested_counts, dtypes):  # pragma: no cover
    return init_tuple_arr(
        bodo.libs.struct_arr_ext.pre_alloc_struct_array(
            n, nested_counts, dtypes, None, None
        )
    )


def pre_alloc_tuple_array_equiv(
    self, scope, equiv_set, loc, args, kws
):  # pragma: no cover
    """Array analysis function for pre_alloc_tuple_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 3 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_tuple_arr_ext_pre_alloc_tuple_array = (
    pre_alloc_tuple_array_equiv
)


@overload(operator.getitem, no_unliteral=True)
def tuple_arr_getitem(arr, ind):
    if not isinstance(arr, TupleArrayType):
        return

    if isinstance(ind, types.Integer):
        # TODO: warning if value is NA?
        func_text = "def impl(arr, ind):\n"
        tup_data = ",".join(
            f"get_data(arr._data)[{i}][ind]" for i in range(len(arr.data))
        )
        func_text += f"  return ({tup_data})\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "get_data": bodo.libs.struct_arr_ext.get_data,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    if (
        is_list_like_index_type(ind)
        and (ind.dtype == types.bool_ or isinstance(ind.dtype, types.Integer))
    ) or isinstance(ind, types.SliceType):
        # other getitem cases return an array, so just call getitem on underlying data array
        def impl_arr(arr, ind):  # pragma: no cover
            return init_tuple_arr(arr._data[ind])

        return impl_arr

    # This should be the only TupleArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for TupleArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def tuple_arr_setitem(arr, ind, val):
    if not isinstance(arr, TupleArrayType):
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    if isinstance(ind, types.Integer):
        n_fields = len(arr.data)
        func_text = "def impl(arr, ind, val):\n"
        func_text += "  data = get_data(arr._data)\n"
        func_text += "  null_bitmap = get_null_bitmap(arr._data)\n"
        func_text += "  set_bit_to_arr(null_bitmap, ind, 1)\n"
        for i in range(n_fields):
            func_text += f"  data[{i}][ind] = val[{i}]\n"

        loc_vars = {}
        exec(
            func_text,
            {
                "get_data": bodo.libs.struct_arr_ext.get_data,
                "get_null_bitmap": bodo.libs.struct_arr_ext.get_null_bitmap,
                "set_bit_to_arr": bodo.libs.int_arr_ext.set_bit_to_arr,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    # other setitem cases set an array, so just call setitem on underlying data array
    def impl_arr(arr, ind, val):
        # TODO: set scalar_to_arr_len
        val = bodo.utils.conversion.coerce_to_array(val, use_nullable_array=True)
        arr._data[ind] = val._data

    return impl_arr


@overload(len, no_unliteral=True)
def overload_tuple_arr_len(A):
    if isinstance(A, TupleArrayType):
        return lambda A: len(A._data)  # pragma: no cover


@overload_attribute(TupleArrayType, "shape")
def overload_tuple_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(TupleArrayType, "dtype")
def overload_tuple_arr_dtype(A):
    return lambda A: np.object_  # pragma: no cover


@overload_attribute(TupleArrayType, "ndim")
def overload_tuple_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(TupleArrayType, "nbytes")
def overload_tuple_arr_nbytes(A):
    return lambda A: A._data.nbytes  # pragma: no cover


@overload_method(TupleArrayType, "copy", no_unliteral=True)
def overload_tuple_arr_copy(A):
    def copy_impl(A):  # pragma: no cover
        return init_tuple_arr(A._data.copy())

    return copy_impl
