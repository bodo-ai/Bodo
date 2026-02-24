import enum
import operator

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
import bodo.pandas as bd
from bodo.utils.typing import (
    NOT_CONSTANT,
    BodoError,
    MetaType,
    check_unsupported_args,
    dtype_to_array_type,
    get_literal_value,
    get_overload_const,
    get_overload_const_bool,
    is_common_scalar_dtype,
    is_iterable_type,
    is_list_like_index_type,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_none,
    is_overload_true,
    is_scalar_type,
    raise_bodo_error,
)


# type for pd.CategoricalDtype objects in Pandas
class PDCategoricalDtype(types.Opaque):
    def __init__(self, categories, elem_type, ordered, data=None, int_type=None):
        # categories can be None since may not be known (e.g. Series.astype("category"))
        self.categories = categories
        # element type is necessary since categories may not be known
        # elem_type may be None if unknown
        self.elem_type = elem_type
        # ordered may be None if unknown
        self.ordered = ordered
        self.data = _get_cat_index_type(elem_type) if data is None else data
        # Parquet dictionary type may not use the minimum possible int data type so
        # we need to set explicitly
        self.int_type = int_type
        name = f"PDCategoricalDtype({self.categories}, {self.elem_type}, {self.ordered}, {self.data}, {self.int_type})"
        super().__init__(name=name)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@typeof_impl.register(pd.CategoricalDtype)
def _typeof_pd_cat_dtype(val, c):
    cats = tuple(val.categories)
    # Using array.dtype instead of typeof(cats[0]) since Interval values are not
    # supported yet (see test_cut)
    elem_type = None if len(cats) == 0 else bodo.typeof(val.categories.values).dtype
    # we set _int_type in gen_column_read() of Parquet read to pass proper type info
    int_type = getattr(val, "_int_type", None)
    return PDCategoricalDtype(
        cats, elem_type, val.ordered, bodo.typeof(val.categories), int_type
    )


def _get_cat_index_type(elem_type):
    """return the Index type that holds "categories" values given the element type"""
    # NOTE assuming data type is string if unknown (TODO: test this possibility)
    elem_type = bodo.types.string_type if elem_type is None else elem_type
    return bodo.utils.typing.get_index_type_from_dtype(elem_type)


@lower_constant(PDCategoricalDtype)
def lower_constant_categorical_type(context, builder, typ, pyval):
    categories = context.get_constant_generic(
        builder, bodo.typeof(pyval.categories), pyval.categories
    )
    ordered = context.get_constant(types.bool_, pyval.ordered)

    return lir.Constant.literal_struct([categories, ordered])


# store data and nulls as regular arrays without payload machineray
# since this struct is immutable (also immutable in Pandas).
# CategoricalArrayType dtype is mutable in pandas. For example,
# Series.cat.categories = [...] can set values, but we can transform it to
# rename_categories() to avoid mutations
@register_model(PDCategoricalDtype)
class PDCategoricalDtypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("categories", fe_type.data),
            ("ordered", types.bool_),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(PDCategoricalDtype, "categories", "categories")
make_attribute_wrapper(PDCategoricalDtype, "ordered", "ordered")


@intrinsic(prefer_literal=True)
def init_cat_dtype(typingctx, categories_typ, ordered_typ, int_type, cat_vals_typ):
    """Create a CategoricalDtype from categories array and ordered flag"""
    assert bodo.hiframes.pd_index_ext.is_index_type(categories_typ), (
        "init_cat_dtype requires index type for categories"
    )
    assert is_overload_constant_bool(ordered_typ), (
        "init_cat_dtype requires constant ordered flag"
    )
    cat_int_type = None if is_overload_none(int_type) else int_type.dtype
    assert is_overload_none(cat_vals_typ) or isinstance(cat_vals_typ, types.TypeRef), (
        "init_cat_dtype requires constant category values"
    )
    cat_vals = (
        None if is_overload_none(cat_vals_typ) else cat_vals_typ.instance_type.meta
    )

    def codegen(context, builder, sig, args):
        categories, ordered, _, _ = args
        cat_dtype = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        cat_dtype.categories = categories
        context.nrt.incref(builder, sig.args[0], categories)
        context.nrt.incref(builder, sig.args[1], ordered)
        cat_dtype.ordered = ordered
        return cat_dtype._getvalue()

    ret_type = PDCategoricalDtype(
        cat_vals,
        categories_typ.dtype,
        is_overload_true(ordered_typ),
        categories_typ,
        cat_int_type,
    )
    return ret_type(categories_typ, ordered_typ, int_type, cat_vals_typ), codegen


@unbox(PDCategoricalDtype)
def unbox_cat_dtype(typ, obj, c):
    """
    Convert a pd.CategoricalDtype object to a native structure.
    """
    cat_dtype = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # unbox obj.ordered flag
    ordered_obj = c.pyapi.object_getattr_string(obj, "ordered")
    cat_dtype.ordered = c.pyapi.to_native_value(types.bool_, ordered_obj).value
    c.pyapi.decref(ordered_obj)

    # unbox obj.categories.values
    categories_index_obj = c.pyapi.object_getattr_string(obj, "categories")
    cat_dtype.categories = c.pyapi.to_native_value(typ.data, categories_index_obj).value
    c.pyapi.decref(categories_index_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(cat_dtype._getvalue(), is_error=is_error)


@box(PDCategoricalDtype)
def box_cat_dtype(typ, val, c):
    """Box PDCategoricalDtype into pandas CategoricalDtype object."""
    cat_dtype = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    # box ordered flag
    ordered_obj = c.pyapi.from_native_value(
        types.bool_, cat_dtype.ordered, c.env_manager
    )
    # box categories data
    c.context.nrt.incref(c.builder, typ.data, cat_dtype.categories)
    categories_obj = c.pyapi.from_native_value(
        typ.data, cat_dtype.categories, c.env_manager
    )
    # call pd.CategoricalDtype()
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)
    dtype_obj = c.pyapi.call_method(
        pd_class_obj, "CategoricalDtype", (categories_obj, ordered_obj)
    )

    c.pyapi.decref(ordered_obj)
    c.pyapi.decref(categories_obj)
    c.pyapi.decref(pd_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return dtype_obj


@overload_attribute(PDCategoricalDtype, "nbytes")
def pd_categorical_nbytes_overload(A):
    return lambda A: A.categories.nbytes + bodo.io.np_io.get_dtype_size(
        types.bool_
    )  # pragma: no cover


# Array of categorical data (similar to Pandas Categorical array)
class CategoricalArrayType(types.ArrayCompatible):
    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__(name=f"CategoricalArrayType({dtype})")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return CategoricalArrayType(self.dtype)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@typeof_impl.register(pd.Categorical)
def _typeof_pd_cat(val, c):
    return CategoricalArrayType(bodo.typeof(val.dtype))


# TODO: use payload to enable mutability?
@register_model(CategoricalArrayType)
class CategoricalArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        int_dtype = get_categories_int_type(fe_type.dtype)
        members = [("dtype", fe_type.dtype), ("codes", types.Array(int_dtype, 1, "C"))]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(CategoricalArrayType, "codes", "codes")
make_attribute_wrapper(CategoricalArrayType, "dtype", "dtype")


@unbox(CategoricalArrayType)
def unbox_categorical_array(typ, val, c):
    """unbox pd.Categorical array to native value"""
    arr_obj = c.pyapi.object_getattr_string(val, "codes")
    dtype = get_categories_int_type(typ.dtype)
    codes = c.pyapi.to_native_value(types.Array(dtype, 1, "C"), arr_obj).value
    c.pyapi.decref(arr_obj)

    dtype_obj = c.pyapi.object_getattr_string(val, "dtype")
    dtype_val = c.pyapi.to_native_value(typ.dtype, dtype_obj).value
    c.pyapi.decref(dtype_obj)

    # create CategoricalArrayType
    cat_arr_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    cat_arr_val.codes = codes
    cat_arr_val.dtype = dtype_val
    return NativeValue(cat_arr_val._getvalue())


@lower_constant(CategoricalArrayType)
def lower_constant_categorical_array(context, builder, typ, pyval):
    """convert constant categorical array value to native value"""

    codes_dtype = get_categories_int_type(typ.dtype)
    codes_arr = context.get_constant_generic(
        builder, types.Array(codes_dtype, 1, "C"), pyval.codes
    )
    cat_dtype = context.get_constant_generic(builder, typ.dtype, pyval.dtype)

    # create CategoricalArrayType
    return lir.Constant.literal_struct([cat_dtype, codes_arr])


def get_categories_int_type(cat_dtype):
    """find smallest integer data type that can represent all categories in 'cat_dtype'"""
    dtype = types.int64

    # Parquet read case provides int data type explicitly (can differ from min possible)
    if cat_dtype.int_type is not None:
        return cat_dtype.int_type

    # if categories are not known upfront, assume worst case int64 for codes
    if cat_dtype.categories is None:
        return types.int64

    n_cats = len(cat_dtype.categories)
    if n_cats < np.iinfo(np.int8).max:
        dtype = types.int8
    elif n_cats < np.iinfo(np.int16).max:
        dtype = types.int16
    elif n_cats < np.iinfo(np.int32).max:
        dtype = types.int32
    return dtype


@box(CategoricalArrayType)
def box_categorical_array(typ, val, c):
    """box native CategoricalArrayType to pd.Categorical array object"""
    dtype = typ.dtype
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)

    # get codes and dtype objects
    int_dtype = get_categories_int_type(dtype)
    cat_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    arr_type = types.Array(int_dtype, 1, "C")
    c.context.nrt.incref(c.builder, arr_type, cat_arr.codes)
    arr_obj = c.pyapi.from_native_value(arr_type, cat_arr.codes, c.env_manager)
    c.context.nrt.incref(c.builder, dtype, cat_arr.dtype)
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
    c.context.nrt.decref(c.builder, typ, val)
    return cat_arr_obj


def _to_readonly(t):
    """convert array or Index type to read-only"""
    from bodo.hiframes.pd_index_ext import (
        DatetimeIndexType,
        NumericIndexType,
        TimedeltaIndexType,
    )

    # TODO: Support the readonly flag on DatetimeIndexType with tz-aware data.
    # TODO(ehsan): add support for other index/array types

    if isinstance(t, CategoricalArrayType):
        return CategoricalArrayType(_to_readonly(t.dtype))

    if isinstance(t, PDCategoricalDtype):
        return PDCategoricalDtype(
            t.categories,
            t.elem_type,
            t.ordered,
            _to_readonly(t.data),
            t.int_type,
        )

    if isinstance(t, types.Array):
        return types.Array(t.dtype, t.ndim, "C", True)

    if isinstance(t, NumericIndexType):
        return NumericIndexType(t.dtype, t.name_typ, _to_readonly(t.data))

    if isinstance(t, (DatetimeIndexType, TimedeltaIndexType)):
        return t.__class__(t.name_typ, _to_readonly(t.data))

    return t


@lower_cast(CategoricalArrayType, CategoricalArrayType)
def cast_cat_arr(context, builder, fromty, toty, val):
    """cast Index array type inside the categorical dtype to read-only since alloc_type
    uses const arrays to create categorical dtypes but the input type may be mutable.
    Index values are immutable anyways.
    see test_groupby.py::test_first_last[categorical_value_df]
    """

    if _to_readonly(toty) == fromty:
        return val

    raise BodoError(f"Cannot cast from {fromty} to {toty}")


def create_cmp_op_overload(op):
    """generate overload for a comparison operator"""

    def overload_cat_arr_cmp(A, other):
        if not isinstance(A, CategoricalArrayType):
            return

        # TODO(ehsan): proper error checking for invalid comparison

        # code for 'other' can be determined ahead of time
        if (
            A.dtype.categories
            and is_literal_type(other)
            and types.unliteral(other) == A.dtype.elem_type
        ):
            val = get_literal_value(other)
            other_idx = (
                list(A.dtype.categories).index(val) if val in A.dtype.categories else -2
            )

            def impl_lit(A, other):  # pragma: no cover
                out_arr = op(
                    bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(A),
                    other_idx,
                )
                return out_arr

            return impl_lit

        def impl(A, other):  # pragma: no cover
            other_idx = get_code_for_value(A.dtype, other)
            out_arr = op(
                bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(A), other_idx
            )
            return out_arr

        return impl

    return overload_cat_arr_cmp


def _install_cmp_ops():
    # install comparison ops: eq, ne
    # TODO(ehsan): support other ops
    for op in [operator.eq, operator.ne]:
        overload_impl = create_cmp_op_overload(op)
        overload(op, inline="always", no_unliteral=True)(overload_impl)


_install_cmp_ops()


@register_jitable
def get_code_for_value(cat_dtype, val):
    """get categorical code for value 'val' by finding its index in categories"""
    # TODO(ehsan): use get_loc when support by arrays
    cat_arr = cat_dtype.categories
    n = len(cat_arr)
    for i in range(n):
        if cat_arr[i] == val:
            return i

    return -2  # return dummy value that doesn't match any categorical code


@overload_method(CategoricalArrayType, "astype", inline="always", no_unliteral=True)
def overload_cat_arr_astype(A, dtype, copy=True, _bodo_nan_to_str=False):
    # If dtype is a string, force it to be a literal
    if dtype == types.unicode_type:
        raise_bodo_error(
            "CategoricalArray.astype(): 'dtype' when passed as string must be a constant value"
        )

    nb_dtype = bodo.utils.typing.parse_dtype(dtype, "CategoricalArray.astype")
    # only supports converting back to original data and string
    if (
        nb_dtype != A.dtype.elem_type and nb_dtype != types.unicode_type
    ):  # pragma: no cover
        raise BodoError(
            f"Converting categorical array {A} to dtype {dtype} not supported yet"
        )

    if nb_dtype == types.unicode_type:

        def impl(A, dtype, copy=True, _bodo_nan_to_str=False):  # pragma: no cover
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(A)
            categories = A.dtype.categories
            n = len(codes)
            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
            for i in numba.parfors.parfor.internal_prange(n):
                s = codes[i]
                if s == -1:
                    if _bodo_nan_to_str:
                        bodo.libs.str_arr_ext.str_arr_setitem_NA_str(out_arr, i)
                    else:
                        bodo.libs.array_kernels.setna(out_arr, i)
                    continue
                out_arr[i] = str(
                    bodo.utils.conversion.unbox_if_tz_naive_timestamp(categories[s])
                )
            return out_arr

        return impl

    arr_type = dtype_to_array_type(nb_dtype)

    def impl(A, dtype, copy=True, _bodo_nan_to_str=False):  # pragma: no cover
        codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(A)
        categories = A.dtype.categories
        n = len(codes)
        out_arr = bodo.utils.utils.alloc_type(n, arr_type, (-1,))
        for i in numba.parfors.parfor.internal_prange(n):
            s = codes[i]
            if s == -1:
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            out_arr[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                categories[s]
            )
        return out_arr

    return impl


# HACK: dummy overload for CategoricalDtype to avoid type inference errors
# TODO: implement dtype properly
@overload(pd.api.types.CategoricalDtype, no_unliteral=True)
@overload(bd.api.types.CategoricalDtype, no_unliteral=True)
def cat_overload_dummy(val_list):
    return lambda val_list: 1  # pragma: no cover


@intrinsic
def init_categorical_array(typingctx, codes, cat_dtype):
    """Create a CategoricalArrayType with codes array (integers) and categories dtype"""
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

    ret_typ = CategoricalArrayType(cat_dtype)
    sig = ret_typ(codes, cat_dtype)
    return sig, codegen


def init_categorical_array_equiv(self, scope, equiv_set, loc, args, kws):
    """out array of init_categorical_array has the same shape as input codes array"""
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_categorical_ext_init_categorical_array = init_categorical_array_equiv


def alloc_categorical_array(n, cat_dtype):  # pragma: no cover
    """Wrapper to the allocator so the int type
    for the codes can be extracted via an overload.
    """


# high-level allocation function for categorical arrays
@overload(alloc_categorical_array, no_unliteral=True)
def _alloc_categorical_array(n, cat_dtype):
    int_dtype = get_categories_int_type(cat_dtype)

    def impl(n, cat_dtype):  # pragma: no cover
        codes = np.empty(n, int_dtype)
        return init_categorical_array(codes, cat_dtype)

    return impl


def alloc_categorical_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_categorical_array() passed to Numba's array analysis
    extension. Assigns output array's size as equivalent to the input size variable.
    """
    assert len(args) == 2 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_categorical_ext_alloc_categorical_array = alloc_categorical_array_equiv


# using a function for getting data to enable extending various analysis
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_categorical_arr_codes(A):
    return lambda A: A.codes  # pragma: no cover


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    """the codes array is kept inside Categorical array so it aliases"""
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_categorical_array", "bodo.hiframes.pd_categorical_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_categorical_arr_codes", "bodo.hiframes.pd_categorical_ext")
] = alias_ext_dummy_func


@overload_method(CategoricalArrayType, "copy", no_unliteral=True)
def cat_arr_copy_overload(arr):
    return lambda arr: init_categorical_array(
        arr.codes.copy(), arr.dtype
    )  # pragma: no cover


def build_replace_dicts(to_replace, value, categories):  # pragma: no cover
    """Helper functions to build arrays used by the replace method. Should support
    the same set of cases as those provided for Series.replace. However for categories
    we need the following 4 things to perform the swaps:
    1. Mappings that will be replaced, for updating categories.
    2. Mappings that will be deleted, for updating categories.
    3. Code mapping updates. When categories are deleted code values may drop and change. This
    value can be an array because the keys are always range(-1, n)
    4. Number of categories deleted.
    We can group 1 and 2 together by filtering out Map[a] -> a and letting
    mapping to yourself serve as a deletion. This results in 3 return values:
    category_dict, codes_arr, num_deleted
    """
    return {}, np.empty(len(categories) + 1), 0


@overload(build_replace_dicts, no_unliteral=True)
def _build_replace_dicts(to_replace, value, categories):
    # Scalar case
    # TODO: replace with something that captures all scalars
    if isinstance(to_replace, types.Number) or to_replace == bodo.types.string_type:

        def impl(to_replace, value, categories):  # pragma: no cover
            return build_replace_dicts([to_replace], value, categories)

        return impl

    # List case with scalar value
    # TODO: replace with explicit checking for to_replace types that are/aren't supported
    else:

        def impl(to_replace, value, categories):  # pragma: no cover
            n = len(categories)
            # map (old category) -> (new category), Only changed categories will be mapped.
            # Deleted categories will map to themselves (to avoid a second map).
            categories_dict = {}
            # map (old codes) -> (new codes)
            # TODO: Allow replacing na
            codes_arr = np.empty(n + 1, np.int64)
            # map (replaced codes) -> (new code in categories) This is before remapping
            # codes to decrement by removed codes.
            replace_codes_dict = {}
            # List of codes that will be deleted. Used for updating code mappings
            delete_codes_list = []
            # map(category) -> (old code value)
            cat_to_code = {}
            for i in range(n):
                cat_to_code[categories[i]] = i
            # Determine which categories are getting remapped/deleted
            for replace_cat in to_replace:
                # Skip replaces with themselves because they won't change
                # the code.
                if replace_cat != value:
                    if replace_cat in cat_to_code:
                        # For deletions update the categories
                        if value in cat_to_code:
                            categories_dict[replace_cat] = replace_cat
                            code_replacee = cat_to_code[replace_cat]
                            replace_codes_dict[code_replacee] = cat_to_code[value]
                            delete_codes_list.append(code_replacee)
                        else:
                            categories_dict[replace_cat] = value
                            cat_to_code[value] = cat_to_code[replace_cat]
            delete_codes = np.sort(np.array(delete_codes_list))
            # Determine how much each code must decrease before constructing
            # final mapping
            decr_value = 0
            decr_counts = []
            for j in range(-1, n):
                while decr_value < len(delete_codes) and j > delete_codes[decr_value]:
                    decr_value += 1
                decr_counts.append(decr_value)
            for k in range(-1, n):
                search_location = k
                if k in replace_codes_dict:
                    search_location = replace_codes_dict[k]
                codes_arr[k + 1] = search_location - decr_counts[search_location + 1]
            return categories_dict, codes_arr, len(delete_codes)

        return impl


@numba.njit
def python_build_replace_dicts(to_replace, value, categories):  # pragma: no cover
    """Jit wrapper to call build_replace_dicts from Python"""
    return build_replace_dicts(to_replace, value, categories)


@register_jitable
def reassign_codes(new_codes_arr, old_codes_arr, codes_map_arr):  # pragma: no cover
    """Helper function to remap codes in replace."""
    for i in range(len(new_codes_arr)):
        new_codes_arr[i] = codes_map_arr[old_codes_arr[i] + 1]


@overload_method(CategoricalArrayType, "replace", inline="always", no_unliteral=True)
def overload_replace(arr, to_replace, value):
    def impl(arr, to_replace, value):  # pragma: no cover
        return bodo.hiframes.pd_categorical_ext.cat_replace(arr, to_replace, value)

    return impl


def cat_replace(arr, to_replace, value):  # pragma: no cover
    # Dummy function for creating a builtin
    return


@overload(cat_replace, no_unliteral=True)
def cat_replace_overload(arr, to_replace, value):
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        to_replace, "CategoricalArray.replace()"
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        value, "CategoricalArray.replace()"
    )
    _ordered = arr.dtype.ordered
    _elem_type = arr.dtype.elem_type

    to_replace_constant = get_overload_const(to_replace)
    value_constant = get_overload_const(value)

    # If we have known categories and constant inputs we can construct a known dtype.
    if (
        arr.dtype.categories is not None
        and to_replace_constant is not NOT_CONSTANT
        and value_constant is not NOT_CONSTANT
    ):
        cats_dict, codes_map_arr, _ = python_build_replace_dicts(
            to_replace_constant, value_constant, arr.dtype.categories
        )
        # If nothing is being changed we can just return a copy of the aray
        if len(cats_dict) == 0:
            return lambda arr, to_replace, value: arr.copy()
        # Otherwise create a new dtype
        cats_list = []
        for cat in arr.dtype.categories:
            if cat in cats_dict:
                new_cat = cats_dict[cat]
                # If it maps to itself this is a deletion. This is because we filtered out
                # all actual attempts to map to itself
                if new_cat != cat:
                    cats_list.append(new_cat)

            else:
                cats_list.append(cat)

        # create the new categorical dtype inside the function instead of passing as
        # constant. This avoids constant lowered Index inside the dtype, which can be
        # slow since it cannot have a dictionary.
        # see https://github.com/bodo-ai/Bodo/pull/3563
        new_categories = bodo.utils.utils.create_categorical_type(
            cats_list, arr.dtype.data.data, _ordered
        )
        new_categories_tup = MetaType(tuple(new_categories))

        # Implementation avoids changing the actual categories and knows the
        # categories must change
        def impl_dtype(arr, to_replace, value):  # pragma: no cover
            new_dtype = init_cat_dtype(
                bodo.utils.conversion.index_from_array(new_categories),
                _ordered,
                None,
                new_categories_tup,
            )
            cat_arr = alloc_categorical_array(len(arr.codes), new_dtype)
            # Use codes_map_arr from compile time
            reassign_codes(cat_arr.codes, arr.codes, codes_map_arr)
            return cat_arr

        return impl_dtype

    _elem_type = arr.dtype.elem_type
    # Handle strings differently until we get an array builder
    if _elem_type == types.unicode_type:

        def impl_str(arr, to_replace, value):  # pragma: no cover
            categories = arr.dtype.categories
            categories_dict, codes_map_arr, num_deleted = build_replace_dicts(
                to_replace, value, categories.values
            )
            if len(categories_dict) == 0:
                return init_categorical_array(
                    arr.codes.copy().astype(np.int64),
                    init_cat_dtype(categories.copy(), _ordered, None, None),
                )
            # If we must edit the categories we need to preallocate a new
            # string array
            n = len(categories)
            new_categories = bodo.libs.str_arr_ext.pre_alloc_string_array(
                n - num_deleted, -1
            )
            # Fill in all the categories in the new array
            new_idx = 0
            for j in range(n):
                old_cat_val = categories[j]
                if old_cat_val in categories_dict:
                    new_cat_val = categories_dict[old_cat_val]
                    # if new == old, its a deletion, used to avoid a second dict
                    if new_cat_val != old_cat_val:
                        new_categories[new_idx] = new_cat_val
                        new_idx += 1
                else:
                    new_categories[new_idx] = old_cat_val
                    new_idx += 1
            cat_arr = alloc_categorical_array(
                len(arr.codes),
                init_cat_dtype(
                    bodo.utils.conversion.index_from_array(new_categories),
                    _ordered,
                    None,
                    None,
                ),
            )
            # Update all of the codes
            reassign_codes(cat_arr.codes, arr.codes, codes_map_arr)
            return cat_arr

        return impl_str

    _arr_type = dtype_to_array_type(_elem_type)

    def impl(arr, to_replace, value):  # pragma: no cover
        categories = arr.dtype.categories
        categories_dict, codes_map_arr, num_deleted = build_replace_dicts(
            to_replace, value, categories.values
        )
        if len(categories_dict) == 0:
            return init_categorical_array(
                arr.codes.copy().astype(np.int64),
                init_cat_dtype(categories.copy(), _ordered, None, None),
            )
        n = len(categories)
        new_categories = bodo.utils.utils.alloc_type(n - num_deleted, _arr_type, None)
        # Fill in all the categories in the new array
        new_idx = 0
        for i in range(n):
            old_cat_val = categories[i]
            if old_cat_val in categories_dict:
                new_cat_val = categories_dict[old_cat_val]
                # if new == old, its a deletion, used to avoid a second dict
                if new_cat_val != old_cat_val:
                    new_categories[new_idx] = new_cat_val
                    new_idx += 1
            else:
                new_categories[new_idx] = old_cat_val
                new_idx += 1
        # Update all of the codes
        cat_arr = alloc_categorical_array(
            len(arr.codes),
            init_cat_dtype(
                bodo.utils.conversion.index_from_array(new_categories),
                _ordered,
                None,
                None,
            ),
        )
        reassign_codes(cat_arr.codes, arr.codes, codes_map_arr)
        return cat_arr

    return impl


@overload(len, no_unliteral=True)
def overload_cat_arr_len(A):
    if isinstance(A, CategoricalArrayType):
        return lambda A: len(A.codes)  # pragma: no cover


@overload_attribute(CategoricalArrayType, "shape")
def overload_cat_arr_shape(A):
    return lambda A: (len(A.codes),)  # pragma: no cover


@overload_attribute(CategoricalArrayType, "ndim")
def overload_cat_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(CategoricalArrayType, "nbytes")
def cat_arr_nbytes_overload(A):
    return lambda A: A.codes.nbytes + A.dtype.nbytes  # pragma: no cover


@register_jitable
def get_label_dict_from_categories(vals):  # pragma: no cover
    """Generates the dictionairy mapping categorical values to their integer code value, from a
    collection of collection of categorical values that may contain dupliicates.
    """
    labels = {}

    curr_ind = 0
    for i in range(len(vals)):
        val = vals[i]
        if val in labels:
            continue
        labels[val] = curr_ind
        curr_ind += 1

    return labels


@register_jitable
def get_label_dict_from_categories_no_duplicates(vals):  # pragma: no cover
    """Generates the dictionairy mapping categorical values to their integer code value, from a
    collection of collection of categorical values containing no dupliicates.
    """
    labels = {}
    for i in range(len(vals)):
        val = bodo.utils.conversion.unbox_if_tz_naive_timestamp(vals[i])
        labels[val] = i

    return labels


# NOTE: not using inline="always" since fix_arr_dtype() fails due to Bodo IR nodes.
# Inlined in Series pass.
@overload(pd.Categorical, no_unliteral=True)
@overload(bd.Categorical, no_unliteral=True)
def pd_categorical_overload(
    values,
    categories=None,
    ordered=None,
    dtype=None,
    fastpath=False,
):
    unsupported_args = {"fastpath": fastpath}
    arg_defaults = {"fastpath": False}
    check_unsupported_args("pd.Categorical", unsupported_args, arg_defaults)

    # categorical dtype is provided
    if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):

        def impl_dtype(
            values, categories=None, ordered=None, dtype=None, fastpath=False
        ):  # pragma: no cover
            data = bodo.utils.conversion.coerce_to_array(values)
            return bodo.utils.conversion.fix_arr_dtype(data, dtype)

        return impl_dtype

    # categories are provided
    if not is_overload_none(categories):
        const_categories = get_overload_const(categories)
        # We can create a type at compile time if we have constant categories and ordered.
        if (
            const_categories is not NOT_CONSTANT
            and get_overload_const(ordered) is not NOT_CONSTANT
        ):
            if is_overload_none(ordered):
                is_ordered = False
            else:
                is_ordered = get_overload_const_bool(ordered)

            # create the new categorical dtype inside the function instead of passing as
            # constant. This avoids constant lowered Index inside the dtype, which can
            # be slow since it cannot have a dictionary.
            # see https://github.com/bodo-ai/Bodo/pull/3563
            new_cats_arr = pd.CategoricalDtype(
                pd.array(const_categories), is_ordered
            ).categories.array
            new_cats_tup = MetaType(tuple(new_cats_arr))

            # If the categories are constant, create the type at compile time.
            def impl_cats_const(
                values, categories=None, ordered=None, dtype=None, fastpath=False
            ):  # pragma: no cover
                data = bodo.utils.conversion.coerce_to_array(values)
                new_dtype = init_cat_dtype(
                    bodo.utils.conversion.index_from_array(new_cats_arr),
                    is_ordered,
                    None,
                    new_cats_tup,
                )
                return bodo.utils.conversion.fix_arr_dtype(data, new_dtype)

            return impl_cats_const

        def impl_cats(
            values, categories=None, ordered=None, dtype=None, fastpath=False
        ):  # pragma: no cover
            ordered = bodo.utils.conversion.false_if_none(ordered)
            data = bodo.utils.conversion.coerce_to_array(values)
            cats = bodo.utils.conversion.convert_to_index(categories)
            cat_dtype = bodo.hiframes.pd_categorical_ext.init_cat_dtype(
                cats, ordered, None, None
            )
            return bodo.utils.conversion.fix_arr_dtype(data, cat_dtype)

        return impl_cats

    else:
        if is_overload_none(ordered):

            def impl_auto(
                values, categories=None, ordered=None, dtype=None, fastpath=False
            ):  # pragma: no cover
                data = bodo.utils.conversion.coerce_to_array(values)
                return bodo.utils.conversion.fix_arr_dtype(data, "category")

            return impl_auto
        # TODO(ehsan): handle ordered case

    raise BodoError(
        f"pd.Categorical(): argument combination not supported yet: {values}, {categories}, {ordered}, {dtype}"
    )


@overload(operator.getitem, no_unliteral=True)
def categorical_array_getitem(arr, ind):
    if not isinstance(arr, CategoricalArrayType):
        return

    # scalar int
    if isinstance(ind, types.Integer):
        # TODO: support returning NA
        def categorical_getitem_impl(arr, ind):  # pragma: no cover
            code = arr.codes[ind]
            # Returns a dummy value if code == -1. The user needs to handle
            # this with isna the same as other arrays.
            return arr.dtype.categories[max(code, 0)]

        return categorical_getitem_impl

    # bool/int/slice arr indexing.
    if is_list_like_index_type(ind) or isinstance(ind, types.SliceType):

        def impl_bool(arr, ind):  # pragma: no cover
            return init_categorical_array(arr.codes[ind], arr.dtype)

        return impl_bool

    # This should be the only CategoricalArrayType implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for CategoricalArray with indexing type {ind} not supported."
    )  # pragma: no cover


class CategoricalMatchingValues(enum.Enum):
    """
    Enum used to determine if two values match.
    MAY_MATCH means the two values may match at runtime,
    but we can't tell at compile time.

    DIFFERENT_TYPES is used if a type examined is not a CategoricalArrayType,
    which should produce a different error message.
    """

    DIFFERENT_TYPES = -1
    DONT_MATCH = 0
    MAY_MATCH = 1
    DO_MATCH = 2


def categorical_arrs_match(arr1, arr2):
    """
    Helper functions that determines if the inputs are matching
    categorical arrays. If either category is None, the types
    can match at runtime (need a runtime check).
    """
    if not (
        isinstance(arr1, CategoricalArrayType)
        and isinstance(arr2, CategoricalArrayType)
    ):
        return CategoricalMatchingValues.DIFFERENT_TYPES
    if arr1.dtype.categories is None or arr2.dtype.categories is None:
        return CategoricalMatchingValues.MAY_MATCH
    return (
        CategoricalMatchingValues.DO_MATCH
        if arr1.dtype.categories == arr2.dtype.categories
        and arr1.dtype.ordered == arr2.dtype.ordered
        else CategoricalMatchingValues.DONT_MATCH
    )


@register_jitable
def cat_dtype_equal(dtype1, dtype2):  # pragma: no cover
    """return True if categorical dtypes are equal
    (checks ordered flag and category values)
    """
    if dtype1.ordered != dtype2.ordered or len(dtype1.categories) != len(
        dtype2.categories
    ):
        return False

    # NOTE: not using (dtype1.categories != dtype1.categories).any() due to bug in
    # Numba's array expr handling (TODO fix)
    arr1 = dtype1.categories.values
    arr2 = dtype2.categories.values
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True


@overload(operator.setitem, no_unliteral=True)
def categorical_array_setitem(arr, ind, val):
    if not isinstance(arr, CategoricalArrayType):
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    # val is scalar RHS that can be assigned.
    is_scalar_match = (
        is_scalar_type(val)
        and is_common_scalar_dtype([types.unliteral(val), arr.dtype.elem_type])
        # Make sure we don't try insert a float into an int. This
        # will pass is_common_scalar_dtype but is incorrect
        and not (
            isinstance(arr.dtype.elem_type, types.Integer)
            and isinstance(val, types.Float)
        )
    )
    # val is an array RHS that can be assigned
    is_arr_rhs = (
        not isinstance(val, CategoricalArrayType)
        and is_iterable_type(val)
        and is_common_scalar_dtype([val.dtype, arr.dtype.elem_type])
        # Make sure we don't try insert a float into an int. This
        # will pass is_common_scalar_dtype but is incorrect
        and not (
            isinstance(arr.dtype.elem_type, types.Integer)
            and isinstance(val.dtype, types.Float)
        )
    )
    # val is a Categorical Array and categories match/can match
    # if they can match we check at compile time.
    cats_match = categorical_arrs_match(arr, val)

    typ_err_msg = f"setitem for CategoricalArrayType of dtype {arr.dtype} with indexing type {ind} received an incorrect 'value' type {val}."
    categories_err_msg = (
        "Cannot set a Categorical with another, without identical categories"
    )

    # scalar case
    if isinstance(ind, types.Integer):
        if not is_scalar_match:
            raise BodoError(typ_err_msg)

        def impl_scalar(arr, ind, val):  # pragma: no cover
            if val not in arr.dtype.categories:
                raise ValueError(
                    "Cannot setitem on a Categorical with a new category, set the categories first"
                )
            code = arr.dtype.categories.get_loc(val)
            arr.codes[ind] = code

        return impl_scalar

    # array of int indices
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):
        if not (
            is_scalar_match
            or is_arr_rhs
            or cats_match != CategoricalMatchingValues.DIFFERENT_TYPES
        ):
            raise BodoError(typ_err_msg)

        if cats_match == CategoricalMatchingValues.DONT_MATCH:
            raise BodoError(categories_err_msg)

        if is_scalar_match:

            def impl_scalar(arr, ind, val):  # pragma: no cover
                if val not in arr.dtype.categories:
                    raise ValueError(
                        "Cannot setitem on a Categorical with a new category, set the categories first"
                    )
                val_code = arr.dtype.categories.get_loc(val)
                n = len(ind)
                for j in range(n):
                    arr.codes[ind[j]] = val_code

            return impl_scalar

        if cats_match == CategoricalMatchingValues.DO_MATCH:

            def impl_arr_ind_mask(arr, ind, val):  # pragma: no cover
                n = len(val.codes)
                for i in range(n):
                    arr.codes[ind[i]] = val.codes[i]

            return impl_arr_ind_mask

        if cats_match == CategoricalMatchingValues.MAY_MATCH:

            def impl_arr_ind_mask(arr, ind, val):  # pragma: no cover
                if not cat_dtype_equal(arr.dtype, val.dtype):
                    raise ValueError(categories_err_msg)
                n = len(val.codes)
                for i in range(n):
                    arr.codes[ind[i]] = val.codes[i]

            return impl_arr_ind_mask

        if is_arr_rhs:

            def impl_arr_ind_mask_cat_values(arr, ind, val):  # pragma: no cover
                n = len(val)
                categories = arr.dtype.categories

                for j in range(n):
                    # Timestamp/Timedelta are stored internally as dt64 but inside the index as
                    # Timestamp and Timedelta
                    new_val = bodo.utils.conversion.unbox_if_tz_naive_timestamp(val[j])
                    if new_val not in categories:
                        raise ValueError(
                            "Cannot setitem on a Categorical with a new category, set the categories first"
                        )
                    code = categories.get_loc(new_val)
                    arr.codes[ind[j]] = code

            return impl_arr_ind_mask_cat_values

    # bool array
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:
        if not (
            is_scalar_match
            or is_arr_rhs
            or cats_match != CategoricalMatchingValues.DIFFERENT_TYPES
        ):
            raise BodoError(typ_err_msg)

        if cats_match == CategoricalMatchingValues.DONT_MATCH:
            raise BodoError(categories_err_msg)

        if is_scalar_match:

            def impl_scalar(arr, ind, val):  # pragma: no cover
                if val not in arr.dtype.categories:
                    raise ValueError(
                        "Cannot setitem on a Categorical with a new category, set the categories first"
                    )
                val_code = arr.dtype.categories.get_loc(val)
                n = len(ind)
                for j in range(n):
                    if ind[j]:
                        arr.codes[j] = val_code

            return impl_scalar

        if cats_match == CategoricalMatchingValues.DO_MATCH:

            def impl_bool_ind_mask(arr, ind, val):  # pragma: no cover
                n = len(ind)
                val_ind = 0
                for i in range(n):
                    if ind[i]:
                        arr.codes[i] = val.codes[val_ind]
                        val_ind += 1

            return impl_bool_ind_mask

        if cats_match == CategoricalMatchingValues.MAY_MATCH:

            def impl_bool_ind_mask(arr, ind, val):  # pragma: no cover
                if not cat_dtype_equal(arr.dtype, val.dtype):
                    raise ValueError(categories_err_msg)
                n = len(ind)
                val_ind = 0
                for i in range(n):
                    if ind[i]:
                        arr.codes[i] = val.codes[val_ind]
                        val_ind += 1

            return impl_bool_ind_mask

        if is_arr_rhs:

            def impl_bool_ind_mask_cat_values(arr, ind, val):  # pragma: no cover
                n = len(ind)
                val_ind = 0
                categories = arr.dtype.categories
                for j in range(n):
                    if ind[j]:
                        # Timestamp/Timedelta are stored internally as dt64 but inside the index as
                        # Timestamp and Timedelta
                        new_val = bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                            val[val_ind]
                        )
                        if new_val not in categories:
                            raise ValueError(
                                "Cannot setitem on a Categorical with a new category, set the categories first"
                            )
                        code = categories.get_loc(new_val)
                        arr.codes[j] = code
                        val_ind += 1

            return impl_bool_ind_mask_cat_values

    # slice case
    if isinstance(ind, types.SliceType):
        if not (
            is_scalar_match
            or is_arr_rhs
            or cats_match != CategoricalMatchingValues.DIFFERENT_TYPES
        ):
            raise BodoError(typ_err_msg)

        if cats_match == CategoricalMatchingValues.DONT_MATCH:
            raise BodoError(categories_err_msg)

        if is_scalar_match:

            def impl_scalar(arr, ind, val):  # pragma: no cover
                if val not in arr.dtype.categories:
                    raise ValueError(
                        "Cannot setitem on a Categorical with a new category, set the categories first"
                    )
                val_code = arr.dtype.categories.get_loc(val)
                slice_ind = numba.cpython.unicode._normalize_slice(ind, len(arr))
                for j in range(slice_ind.start, slice_ind.stop, slice_ind.step):
                    arr.codes[j] = val_code

            return impl_scalar

        if cats_match == CategoricalMatchingValues.DO_MATCH:

            def impl_arr(arr, ind, val):  # pragma: no cover
                arr.codes[ind] = val.codes

            return impl_arr

        if cats_match == CategoricalMatchingValues.MAY_MATCH:

            def impl_arr(arr, ind, val):  # pragma: no cover
                if not cat_dtype_equal(arr.dtype, val.dtype):
                    raise ValueError(categories_err_msg)
                arr.codes[ind] = val.codes

            return impl_arr

        if is_arr_rhs:

            def impl_slice_cat_values(arr, ind, val):  # pragma: no cover
                categories = arr.dtype.categories

                slice_ind = numba.cpython.unicode._normalize_slice(ind, len(arr))
                val_ind = 0
                for j in range(slice_ind.start, slice_ind.stop, slice_ind.step):
                    # Timestamp/Timedelta are stored internally as dt64 but inside the index as
                    # Timestamp and Timedelta
                    new_val = bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                        val[val_ind]
                    )
                    if new_val not in categories:
                        raise ValueError(
                            "Cannot setitem on a Categorical with a new category, set the categories first"
                        )
                    code = categories.get_loc(new_val)
                    arr.codes[j] = code
                    val_ind += 1

            return impl_slice_cat_values

    # This should be the only CategoricalArrayType implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"setitem for CategoricalArrayType with indexing type {ind} not supported."
    )  # pragma: no cover
