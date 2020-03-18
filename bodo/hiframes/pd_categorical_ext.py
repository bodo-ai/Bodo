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
from numba import cgutils, types


# type for pd.CategoricalDtype objects in Pandas
class PDCategoricalDtype(types.Opaque):
    def __init__(self, _categories):
        self.categories = _categories
        name = "PDCategoricalDtype({})".format(self.categories)
        super(PDCategoricalDtype, self).__init__(name=name)


@typeof_impl.register(pd.CategoricalDtype)
def _typeof_pd_cat_dtype(val, c):
    return PDCategoricalDtype(val.categories.to_list())


register_model(PDCategoricalDtype)(models.OpaqueModel)


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


@register_model(CategoricalArray)
class CategoricalArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        int_dtype = get_categories_int_type(fe_type.dtype)
        members = [("codes", types.Array(int_dtype, 1, "C"))]
        super(CategoricalArrayModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(CategoricalArray, "codes", "_codes")


@unbox(CategoricalArray)
def unbox_categorical_array(typ, val, c):
    arr_obj = c.pyapi.object_getattr_string(val, "codes")
    dtype = get_categories_int_type(typ.dtype)
    codes = c.pyapi.to_native_value(types.Array(dtype, 1, "C"), arr_obj).value
    c.pyapi.decref(arr_obj)

    # create CategoricalArray
    cat_arr_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    cat_arr_val.codes = codes
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
    dtype = typ.dtype
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    # categories list e.g. ['A', 'B', 'C']
    item_objs = _get_cat_obj_items(dtype.categories, c)
    n = len(item_objs)
    list_obj = c.pyapi.list_new(c.context.get_constant(types.intp, n))
    for i in range(n):
        idx = c.context.get_constant(types.intp, i)
        c.pyapi.incref(item_objs[i])
        c.pyapi.list_setitem(list_obj, idx, item_objs[i])
    # TODO: why does list_pack crash for test_csv_cat2?
    # list_obj = c.pyapi.list_pack(item_objs)

    # call pd.api.types.CategoricalDtype(['A', 'B', 'C'])
    # api_obj = c.pyapi.object_getattr_string(pd_class_obj, "api")
    # types_obj = c.pyapi.object_getattr_string(api_obj, "types")
    # pd_dtype = c.pyapi.call_method(types_obj, "CategoricalDtype", (list_obj,))
    # c.pyapi.decref(api_obj)
    # c.pyapi.decref(types_obj)

    int_dtype = get_categories_int_type(dtype)
    codes = cgutils.create_struct_proxy(typ)(c.context, c.builder, val).codes
    arr = c.pyapi.from_native_value(
        types.Array(int_dtype, 1, "C"), codes, c.env_manager
    )

    pdcat_cls_obj = c.pyapi.object_getattr_string(pd_class_obj, "Categorical")
    cat_arr = c.pyapi.call_method(pdcat_cls_obj, "from_codes", (arr, list_obj))
    c.pyapi.decref(pdcat_cls_obj)
    c.pyapi.decref(arr)
    c.pyapi.decref(list_obj)
    for obj in item_objs:
        c.pyapi.decref(obj)

    c.pyapi.decref(pd_class_obj)
    return cat_arr


# TODO: handle all ops
@overload(operator.eq)
def overload_cat_arr_eq_str(A, other):
    if isinstance(A, CategoricalArray) and isinstance(other, types.StringLiteral):
        other_idx = list(A.dtype.categories).index(other.literal_value)

        def impl(A, other):  # pragma: no cover
            out_arr = A._codes == other_idx
            return out_arr

        return impl


@lower_cast(CategoricalArray, types.Array)
def cast_cat_arr(context, builder, fromty, toty, val):
    return val


def _get_cat_obj_items(categories, c):
    assert len(categories) > 0
    val = categories[0]
    if isinstance(val, str):
        return [c.pyapi.string_from_constant_string(item) for item in categories]

    dtype = numba.typeof(val)
    return [c.box(dtype, c.context.get_constant(dtype, item)) for item in categories]


# HACK: dummy overload for CategoricalDtype to avoid type inference errors
# TODO: implement dtype properly
@overload(pd.api.types.CategoricalDtype)
def cat_overload_dummy(val_list):
    return lambda val_list: 1


@intrinsic
def init_categorical_array(typingctx, codes, cat_dtype=None):
    """Create a CategoricalArray with codes array (integers) and categories dtype
    """
    assert isinstance(codes, types.Array) and isinstance(codes.dtype, types.Integer)

    def codegen(context, builder, signature, args):
        data_val = args[0]
        # create cat_arr struct and store values
        cat_arr = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        cat_arr.codes = data_val

        # increase refcount of stored array
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)

        return cat_arr._getvalue()

    ret_typ = CategoricalArray(cat_dtype)
    sig = ret_typ(codes, cat_dtype)
    return sig, codegen


@overload_method(CategoricalArray, "copy")
def cat_arr_copy_overload(arr):
    return lambda arr: init_categorical_array(arr._codes.copy(), arr.dtype)


@overload(len)
def overload_cat_arr_len(A):
    if isinstance(A, CategoricalArray):
        return lambda A: len(A._codes)


@overload_attribute(CategoricalArray, "shape")
def overload_cat_arr_shape(A):
    return lambda A: (len(A._codes),)


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


@overload_attribute(CategoricalArray, "dtype")
def overload_cat_arr_dtype(A):
    dtype = A.dtype
    return lambda A: init_cat_dtype(dtype)
