# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import numpy as np
import pandas as pd
import numba
from numba.extending import (
    box,
    unbox,
    typeof_impl,
    register_model,
    make_attribute_wrapper,
    models,
    NativeValue,
    lower_builtin,
    lower_cast,
    overload,
    type_callable,
    overload_method,
    overload_attribute,
    intrinsic,
)
from numba.core import cgutils, types
from llvmlite import ir as lir
import bodo


# type for pd.CategoricalDtype objects in Pandas
class PDCategoricalDtype(types.Opaque):
    def __init__(self, categories, elem_type, ordered):
        # categories can be None since may not be known (e.g. Series.astype("category"))
        self.categories = categories
        # element type is necessary since categories may not be known
        # elem_type may be None if unknown
        self.elem_type = elem_type
        # ordered may be None if unknown
        self.ordered = ordered
        name = "PDCategoricalDtype({}, {}, {})".format(
            self.categories, self.elem_type, self.ordered
        )
        super(PDCategoricalDtype, self).__init__(name=name)


@typeof_impl.register(pd.CategoricalDtype)
def _typeof_pd_cat_dtype(val, c):
    cats = val.categories.to_list()
    elem_type = None if len(cats) == 0 else bodo.typeof(cats[0])
    return PDCategoricalDtype(tuple(cats), elem_type, val.ordered)


def _get_cat_arr_type(elem_type):
    """return the array type that holds "categories" values give the element type
    """
    # NOTE assuming data type is string if unknown (TODO: test this possibility)
    return (
        bodo.string_array_type
        if elem_type is None
        else bodo.hiframes.pd_series_ext._get_series_array_type(elem_type)
    )


# store data and nulls as regular arrays without payload machineray
# since this struct is immutable (also immutable in Pandas).
# CategoricalArray dtype is mutable in pandas. For example,
# Series.cat.categories = [...] can set values, but we can transform it to
# rename_categories() to avoid mutations
@register_model(PDCategoricalDtype)
class PDCategoricalDtypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        arr_type = _get_cat_arr_type(fe_type.elem_type)
        members = [
            ("ordered", types.bool_),
            ("categories", arr_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@unbox(PDCategoricalDtype)
def unbox_cat_dtype(typ, obj, c):
    """
    Convert a pd.CategoricalDtype object to a native structure.
    """
    cat_dtype = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # unbox obj.ordered flag
    ordered_obj = c.pyapi.object_getattr_string(obj, "ordered")
    # unbox bool similar to Numba's bool unboxing
    istrue = c.pyapi.object_istrue(ordered_obj)
    zero = lir.Constant(istrue.type, 0)
    cat_dtype.ordered = c.builder.icmp_signed("!=", istrue, zero)
    c.pyapi.decref(ordered_obj)

    # unbox obj.categories.values
    categories_index_obj = c.pyapi.object_getattr_string(obj, "categories")
    categories_arr_obj = c.pyapi.object_getattr_string(categories_index_obj, "values")
    arr_type = _get_cat_arr_type(typ.elem_type)
    cat_dtype.categories = c.pyapi.to_native_value(arr_type, categories_arr_obj).value
    c.pyapi.decref(categories_index_obj)
    c.pyapi.decref(categories_arr_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(cat_dtype._getvalue(), is_error=is_error)


@box(PDCategoricalDtype)
def box_cat_dtype(typ, val, c):
    """Box PDCategoricalDtype into pandas CategoricalDtype object.
    """
    cat_dtype = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    # box ordered flag
    ordered_obj = c.pyapi.from_native_value(
        types.bool_, cat_dtype.ordered, c.env_manager
    )
    # box categories data
    arr_type = _get_cat_arr_type(typ.elem_type)
    categories_obj = c.pyapi.from_native_value(
        arr_type, cat_dtype.categories, c.env_manager
    )
    # call pd.CategoricalDtype()
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype_obj = c.pyapi.call_method(
        pd_class_obj, "CategoricalDtype", (categories_obj, ordered_obj)
    )

    c.pyapi.decref(ordered_obj)
    c.pyapi.decref(categories_obj)
    c.pyapi.decref(pd_class_obj)
    return dtype_obj


# Array of categorical data (similar to Pandas Categorical array)
class CategoricalArray(types.ArrayCompatible):
    def __init__(self, dtype):
        self.dtype = dtype
        super(CategoricalArray, self).__init__(
            name="CategoricalArray({})".format(dtype)
        )

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")


@typeof_impl.register(pd.Categorical)
def _typeof_pd_cat(val, c):
    return CategoricalArray(bodo.typeof(val.dtype))


# TODO: use payload to enable mutability?
@register_model(CategoricalArray)
class CategoricalArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        int_dtype = get_categories_int_type(fe_type.dtype)
        members = [("dtype", fe_type.dtype), ("codes", types.Array(int_dtype, 1, "C"))]
        super(CategoricalArrayModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(CategoricalArray, "codes", "codes")
make_attribute_wrapper(CategoricalArray, "dtype", "dtype")


@unbox(CategoricalArray)
def unbox_categorical_array(typ, val, c):
    """unbox pd.Categorical array to native value
    """
    arr_obj = c.pyapi.object_getattr_string(val, "codes")
    dtype = get_categories_int_type(typ.dtype)
    codes = c.pyapi.to_native_value(types.Array(dtype, 1, "C"), arr_obj).value
    c.pyapi.decref(arr_obj)

    dtype_obj = c.pyapi.object_getattr_string(val, "dtype")
    dtype_val = c.pyapi.to_native_value(typ.dtype, dtype_obj).value
    c.pyapi.decref(dtype_obj)

    # create CategoricalArray
    cat_arr_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    cat_arr_val.codes = codes
    cat_arr_val.dtype = dtype_val
    return NativeValue(cat_arr_val._getvalue())


def get_categories_int_type(cat_dtype):
    dtype = types.int64
    n_cats = len(cat_dtype.categories)
    if n_cats < np.iinfo(np.int8).max:
        dtype = types.int8
    elif n_cats < np.iinfo(np.int16).max:
        dtype = types.int16
    elif n_cats < np.iinfo(np.int32).max:
        dtype = types.int32
    return dtype


@box(CategoricalArray)
def box_categorical_array(typ, val, c):
    """box native CategoricalArray to pd.Categorical array object
    """
    dtype = typ.dtype
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    # get codes and dtype objects
    int_dtype = get_categories_int_type(dtype)
    cat_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    arr_obj = c.pyapi.from_native_value(
        types.Array(int_dtype, 1, "C"), cat_arr.codes, c.env_manager
    )
    dtype_obj = c.pyapi.from_native_value(dtype, cat_arr.dtype, c.env_manager)
    none_obj = c.pyapi.borrow_none()  # no need to decref

    # call pd.Categorical.from_codes()
    pdcat_cls_obj = c.pyapi.object_getattr_string(pd_class_obj, "Categorical")
    cat_arr_obj = c.pyapi.call_method(
        pdcat_cls_obj, "from_codes", (arr_obj, none_obj, none_obj, dtype_obj)
    )

    c.pyapi.decref(pdcat_cls_obj)
    c.pyapi.decref(arr_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(pd_class_obj)
    return cat_arr_obj


# TODO: handle all ops
@overload(operator.eq, no_unliteral=True)
def overload_cat_arr_eq_str(A, other):
    if isinstance(A, CategoricalArray) and isinstance(other, types.StringLiteral):
        other_idx = list(A.dtype.categories).index(other.literal_value)

        def impl(A, other):  # pragma: no cover
            out_arr = A.codes == other_idx
            return out_arr

        return impl


# HACK: dummy overload for CategoricalDtype to avoid type inference errors
# TODO: implement dtype properly
@overload(pd.api.types.CategoricalDtype, no_unliteral=True)
def cat_overload_dummy(val_list):
    return lambda val_list: 1


@intrinsic
def init_categorical_array(typingctx, codes, cat_dtype=None):
    """Create a CategoricalArray with codes array (integers) and categories dtype
    """
    assert isinstance(codes, types.Array) and isinstance(codes.dtype, types.Integer)

    def codegen(context, builder, signature, args):
        data_val, dtype_val = args
        # create cat_arr struct and store values
        cat_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        cat_arr.codes = data_val
        cat_arr.dtype = dtype_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], dtype_val)

        return cat_arr._getvalue()

    ret_typ = CategoricalArray(cat_dtype)
    sig = ret_typ(codes, cat_dtype)
    return sig, codegen


@overload_method(CategoricalArray, "copy", no_unliteral=True)
def cat_arr_copy_overload(arr):
    return lambda arr: init_categorical_array(arr.codes.copy(), arr.dtype)


@overload(len, no_unliteral=True)
def overload_cat_arr_len(A):
    if isinstance(A, CategoricalArray):
        return lambda A: len(A.codes)


@overload_attribute(CategoricalArray, "shape")
def overload_cat_arr_shape(A):
    return lambda A: (len(A.codes),)


@overload_attribute(CategoricalArray, "ndim")
def overload_cat_arr_ndim(A):
    return lambda A: 1


@intrinsic
def init_cat_dtype(typingctx, cat_dtype=None):
    """Create a dummy value for CategoricalDtype
    """

    def codegen(context, builder, signature, args):
        return context.get_dummy_value()

    sig = cat_dtype.instance_type(cat_dtype)
    return sig, codegen
