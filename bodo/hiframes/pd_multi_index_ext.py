"""Support for MultiIndex type of Pandas"""

import operator

import numba
import pandas as pd
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.utils.conversion import ensure_contig_if_np
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    dtype_to_array_type,
    get_val_type_maybe_str_literal,
    is_overload_none,
)


# NOTE: minimal MultiIndex support that just stores the index arrays without factorizing
# the data into `levels` and `codes`
# TODO: support factorizing similar to pd.core.algorithms._factorize_array
class MultiIndexType(types.ArrayCompatible):
    """type class for pd.MultiIndex object"""

    def __init__(self, array_types, names_typ=None, name_typ=None):
        # NOTE: store array types instead of just dtypes since we currently store whole
        # arrays
        # NOTE: array_types and names_typ should be tuples of types
        names_typ = (types.none,) * len(array_types) if names_typ is None else names_typ
        # name is stored in MultiIndex for compatibility witn Index (not always used)
        name_typ = types.none if name_typ is None else name_typ
        self.array_types = array_types
        self.names_typ = names_typ
        self.name_typ = name_typ
        super().__init__(name=f"MultiIndexType({array_types}, {names_typ}, {name_typ})")

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return MultiIndexType(self.array_types, self.names_typ, self.name_typ)

    @property
    def nlevels(self):
        return len(self.array_types)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


# NOTE: just storing the arrays. TODO: store `levels` and `codes`
# even though `name` attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(MultiIndexType)
class MultiIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Tuple(fe_type.array_types)),
            ("names", types.Tuple(fe_type.names_typ)),
            ("name", fe_type.name_typ),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(MultiIndexType, "data", "_data")
make_attribute_wrapper(MultiIndexType, "names", "_names")
make_attribute_wrapper(MultiIndexType, "name", "_name")


@typeof_impl.register(pd.MultiIndex)
def typeof_multi_index(val, c):
    # using array type inference
    # TODO: avoid using .values if possible, since behavior of .values may change
    array_types = tuple(numba.typeof(val.levels[i].values) for i in range(val.nlevels))
    return MultiIndexType(
        array_types,
        tuple(get_val_type_maybe_str_literal(v) for v in val.names),
        numba.typeof(val.name),
    )


@box(MultiIndexType)
def box_multi_index(typ, val, c):
    """box into pd.MultiIndex object. using `from_arrays` since we are just storing
    arrays currently. TODO: support `levels` and `codes`
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module(mod_name)
    multi_index_class_obj = c.pyapi.object_getattr_string(class_obj, "MultiIndex")

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    # incref since boxing functions steal a reference
    c.context.nrt.incref(c.builder, types.Tuple(typ.array_types), index_val.data)
    data_obj = c.pyapi.from_native_value(
        types.Tuple(typ.array_types), index_val.data, c.env_manager
    )
    c.context.nrt.incref(c.builder, types.Tuple(typ.names_typ), index_val.names)
    names_obj = c.pyapi.from_native_value(
        types.Tuple(typ.names_typ), index_val.names, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.name_typ, index_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    sortorder_obj = c.pyapi.borrow_none()
    index_obj = c.pyapi.call_method(
        multi_index_class_obj, "from_arrays", (data_obj, sortorder_obj, names_obj)
    )
    c.pyapi.object_setattr_string(index_obj, "name", name_obj)

    c.pyapi.decref(data_obj)
    c.pyapi.decref(names_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(multi_index_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@unbox(MultiIndexType)
def unbox_multi_index(typ, val, c):
    """ubox pd.MultiIndex object into native representation.
    using `get_level_values()` to get arrays out since we are just storing
    arrays currently. TODO: support `levels` and `codes`
    """
    data_arrs = []
    # save array objects to decref later since array may be created on demand and
    # cleaned up in Pandas
    arr_objs = []
    for i in range(typ.nlevels):
        # generate val.get_level_values(i)
        i_obj = c.pyapi.unserialize(c.pyapi.serialize_object(i))
        level_obj = c.pyapi.call_method(val, "get_level_values", (i_obj,))
        array_obj = c.pyapi.object_getattr_string(level_obj, "values")
        c.pyapi.decref(level_obj)
        c.pyapi.decref(i_obj)
        data_arr = c.pyapi.to_native_value(typ.array_types[i], array_obj).value
        data_arrs.append(data_arr)
        arr_objs.append(array_obj)

    # set data, names and name attributes
    # if array types are uniform, LLVM ArrayType should be used,
    # otherwise, LiteralStructType is needed
    if isinstance(types.Tuple(typ.array_types), types.UniTuple):
        data = cgutils.pack_array(c.builder, data_arrs)
    else:
        data = cgutils.pack_struct(c.builder, data_arrs)
    # names = tuple(val.names)
    names_obj = c.pyapi.object_getattr_string(val, "names")
    tuple_class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tuple))
    names_tup_obj = c.pyapi.call_function_objargs(tuple_class_obj, (names_obj,))
    names = c.pyapi.to_native_value(types.Tuple(typ.names_typ), names_tup_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.names = names
    index_val.name = name

    for array_obj in arr_objs:
        c.pyapi.decref(array_obj)

    c.pyapi.decref(names_obj)
    c.pyapi.decref(tuple_class_obj)
    c.pyapi.decref(names_tup_obj)
    c.pyapi.decref(name_obj)
    return NativeValue(index_val._getvalue())


def from_product_error_checking(iterables, sortorder, names):
    fname = "pandas.MultiIndex.from_product"
    unsupported_args = {"sortorder": sortorder}
    arg_defaults = {"sortorder": None}
    check_unsupported_args(
        fname,
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )
    if not (is_overload_none(names) or isinstance(names, types.BaseTuple)):
        raise BodoError(f"{fname}: names must be None or a tuple.")
    elif not isinstance(iterables, types.BaseTuple):
        raise BodoError(f"{fname}: iterables must be a tuple.")
    elif not is_overload_none(names) and len(iterables) != len(names):
        raise BodoError(f"{fname}: iterables and names must be of the same length.")


def from_product(iterable, sortorder=None, names=None):  # pragma: no cover
    """Overloaded in from_product_overload"""


@overload(from_product)
def from_product_overload(iterables, sortorder=None, names=None):
    from_product_error_checking(iterables, sortorder, names)
    # Convert to array type to match unboxing
    array_types = tuple(dtype_to_array_type(iterable.dtype) for iterable in iterables)
    if is_overload_none(names):
        names_typ = tuple([types.none] * len(iterables))
    else:
        names_typ = names.types
    multiindex_type = MultiIndexType(array_types, names_typ)
    t_name = f"from_product_multiindex{numba.core.ir_utils.next_label()}"
    setattr(types, t_name, multiindex_type)
    func_text = f"""
def impl(iterables, sortorder=None, names=None):
    with bodo.objmode(mi='{t_name}'):
        mi = pd.MultiIndex.from_product(iterables, names=names)
    return mi
"""
    loc_vars = {}
    exec(func_text, {"pd": pd, "bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@intrinsic(prefer_literal=True)
def init_multi_index(typingctx, data, names, name=None):
    """Create a MultiIndex with provided data, names and name values."""
    name = types.none if name is None else name
    # recreate Tuple type to make sure UniTuple is created if types are homogeneous
    # instead of regular Tuple
    # happens in gatherv() implementation of MultiIndex for some reason
    names = types.Tuple(names.types)

    def codegen(context, builder, signature, args):
        data_val, names_val, name_val = args
        # create multi_index struct and store values
        multi_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        multi_index.data = data_val
        multi_index.names = names_val
        multi_index.name = name_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], names_val)
        context.nrt.incref(builder, signature.args[2], name_val)

        return multi_index._getvalue()

    ret_typ = MultiIndexType(data.types, names.types, name)
    return ret_typ(data, names, name), codegen


@overload(len, no_unliteral=True)
def overload_len_pd_multiindex(A):
    if isinstance(A, MultiIndexType):
        return lambda A: len(A._data[0])  # pragma: no cover


@overload(operator.getitem, no_unliteral=True)
def overload_multi_index_getitem(I, ind):
    if not isinstance(I, MultiIndexType):
        return

    # TODO(ehsan): scalar indexing
    if not isinstance(ind, types.Integer):
        n_fields = len(I.array_types)
        func_text = "def impl(I, ind):\n"
        func_text += "  data = I._data\n"
        # ensure_contig_if_np is need for distributed slices to prevent "A" vs
        # "C" mismatch.
        func_text += "  return init_multi_index(({},), I._names, I._name)\n".format(
            ", ".join(f"ensure_contig_if_np(data[{i}][ind])" for i in range(n_fields))
        )
        loc_vars = {}
        exec(
            func_text,
            {
                "init_multi_index": init_multi_index,
                "ensure_contig_if_np": ensure_contig_if_np,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl


@lower_builtin(operator.is_, MultiIndexType, MultiIndexType)
def multi_index_is(context, builder, sig, args):  # pragma: no cover
    aty, bty = sig.args
    if aty != bty:  # pragma: no cover
        return cgutils.false_bit

    def index_is_impl(a, b):  # pragma: no cover
        return a._data is b._data and a._names is b._names and a._name is b._name

    return context.compile_internal(builder, index_is_impl, sig, args)
