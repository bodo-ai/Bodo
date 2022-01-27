# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Table data type for storing dataframe column arrays. Supports storing many columns
(e.g. >10k) efficiently.
"""
import operator
from collections import defaultdict

import numba
import numpy as np
import pandas as pd
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.cpython.listobj import ListInstance
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
from numba.np.arrayobj import _getitem_array_single_int
from numba.parfors.array_analysis import ArrayAnalysis

from bodo.utils.cg_helpers import is_ll_eq
from bodo.utils.typing import (
    BodoError,
    get_overload_const_int,
    is_list_like_index_type,
    is_overload_constant_int,
)


class Table:
    """basic table which is just a list of arrays
    Python definition needed since CSV reader passes arrays from objmode
    """

    def __init__(self, arrs, usecols=None, num_arrs=-1):
        """
        Constructor for a Python Table.
        arrs is a list of arrays to store in the table.
        usecols is a sorted array of indices for each array.
        num_arrs is used to append trailing NULLs.

        For example if
            arrs = [arr0, arr1, arr2]
            usecols = [1, 2, 4]
            num_arrs = 8

        Then logically the table consists of
            [NULL, arr0, arr1, NULL, arr2, NULL, NULL, NULL]

        If usecols is not provided then there are no gaps. Either
        both usecols and num_arrs must be provided or neither must
        be provided.

        Maintaining the existing order ensures each array will be
        inserted in the expected location from typing during unboxing.

        For a more complete discussion on why these changes are needed, see:
        https://bodo.atlassian.net/wiki/spaces/B/pages/921042953/Table+Structure+with+Dead+Columns
        """
        if usecols is not None:
            assert num_arrs != -1, "num_arrs must be provided if usecols is not None"
            # If usecols is provided we need to place everything in the
            # correct index.
            j = 0
            arr_list = []
            for i in range(usecols[-1] + 1):
                if i == usecols[j]:
                    arr_list.append(arrs[j])
                    j += 1
                else:
                    # Append Nones so the offsets don't change in the type.
                    arr_list.append(None)
            # Add any trailing NULLs
            for _ in range(usecols[-1] + 1, num_arrs):
                arr_list.append(None)
            self.arrays = arr_list
        else:
            self.arrays = arrs
        # for debugging purposes (enables adding print(t_arg.block_0) in unittests
        # which are called in python too)
        self.block_0 = arrs

    def __eq__(self, other):
        return isinstance(other, Table) and other.arrays == self.arrays

    def __str__(self) -> str:
        return str(self.arrays)

    def to_pandas(self, index=None):
        """convert table to a DataFrame (with column names just a range of numbers)"""
        n_cols = len(self.arrays)
        data = dict(zip(range(n_cols), self.arrays))
        df = pd.DataFrame(data, index)
        return df


class TableType(types.ArrayCompatible):
    """Bodo Table type that stores column arrays for dataframes.
    Arrays of the same type are stored in the same "block" (kind of similar to Pandas).
    This allows for loop generation for columns of same type instead of generating code
    for each column (important for dataframes with many columns).
    """

    def __init__(self, arr_types, has_runtime_cols=False):
        self.arr_types = arr_types
        self.has_runtime_cols = has_runtime_cols

        # block number for each array in arr_types
        block_nums = []
        # offset within block for each array in arr_types
        block_offsets = []
        # block number for each array type
        type_to_blk = {}
        # current number of arrays in the block
        blk_curr_ind = defaultdict(int)
        # indices of arrays in arr_types for each block
        block_to_arr_ind = defaultdict(list)
        # We only don't have mapping information if
        # columns are only known at runtime.
        if not has_runtime_cols:
            for i, t in enumerate(arr_types):
                if t not in type_to_blk:
                    type_to_blk[t] = len(type_to_blk)
                blk = type_to_blk[t]
                block_nums.append(blk)
                block_offsets.append(blk_curr_ind[blk])
                blk_curr_ind[blk] += 1
                block_to_arr_ind[blk].append(i)
        else:
            # Verify that with runtime columns arr_types is just a tuple
            # with a single type.
            assert (
                isinstance(arr_types, tuple) and len(arr_types) == 1
            ), "Mixing columns known at compile time with number of columns determined at runtime is not yet supported"

        self.block_nums = block_nums
        self.block_offsets = block_offsets
        self.type_to_blk = type_to_blk
        self.block_to_arr_ind = block_to_arr_ind
        super(TableType, self).__init__(
            name=f"TableType({arr_types}, {has_runtime_cols})"
        )

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 2, "C")

    @property
    def key(self):
        return self.arr_types, self.has_runtime_cols

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@typeof_impl.register(Table)
def typeof_table(val, c):
    return TableType(tuple(numba.typeof(arr) for arr in val.arrays))


@register_model(TableType)
class TableTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # store a list of arrays for each block of same type column arrays
        if fe_type.has_runtime_cols:
            members = [("block_0", types.List(fe_type.arr_types[0]))]
        else:
            members = [
                (f"block_{blk}", types.List(t))
                for t, blk in fe_type.type_to_blk.items()
            ]
        # parent df object if result of df unbox, used for unboxing arrays lazily
        # NOTE: Table could be result of set_table_data() and therefore different than
        # the parent (have more columns and/or different types). However, NULL arrays
        # still have the same column index in the parent df for correct unboxing.
        members.append(("parent", types.pyobject))
        # Keep track of the length in the struct directly.
        members.append(("len", types.int64))
        super(TableTypeModel, self).__init__(dmm, fe_type, members)


# for debugging purposes (a table may not have a block)
make_attribute_wrapper(TableType, "block_0", "block_0")
make_attribute_wrapper(TableType, "len", "_len")


@unbox(TableType)
def unbox_table(typ, val, c):
    """unbox Table into native blocks of arrays"""
    arrs_obj = c.pyapi.object_getattr_string(val, "arrays")
    table = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    table.parent = cgutils.get_null_value(table.parent.type)

    none_obj = c.pyapi.make_none()

    zero = c.context.get_constant(types.int64, 0)
    len_ptr = cgutils.alloca_once_value(c.builder, zero)

    # generate code for each block (allows generating a loop for same type arrays)
    # unbox arrays into a list of arrays in table
    for t, blk in typ.type_to_blk.items():
        n_arrs = c.context.get_constant(types.int64, len(typ.block_to_arr_ind[blk]))
        # not using allocate() since its exception causes calling convention error
        _, out_arr_list = ListInstance.allocate_ex(
            c.context, c.builder, types.List(t), n_arrs
        )
        out_arr_list.size = n_arrs
        # lower array of array indices for block to use within the loop
        # using array since list doesn't have constant lowering
        arr_inds = c.context.make_constant_array(
            c.builder,
            types.Array(types.int64, 1, "C"),
            # On windows np.array defaults to the np.int32 for integers.
            # As a result, we manually specify int64 during the array
            # creation to keep the lowered constant consistent with the
            # expected type.
            np.array(typ.block_to_arr_ind[blk], dtype=np.int64),
        )
        arr_inds_struct = c.context.make_array(types.Array(types.int64, 1, "C"))(
            c.context, c.builder, arr_inds
        )
        with cgutils.for_range(c.builder, n_arrs) as loop:
            i = loop.index
            # get array index in "arrays" list and unbox array
            arr_ind = _getitem_array_single_int(
                c.context,
                c.builder,
                types.int64,
                types.Array(types.int64, 1, "C"),
                arr_inds_struct,
                i,
            )
            # If the value is not null the nstore the array.
            arr_ind_obj = c.pyapi.long_from_longlong(arr_ind)
            arr_obj = c.pyapi.object_getitem(arrs_obj, arr_ind_obj)

            is_none_val = c.builder.icmp_unsigned("==", arr_obj, none_obj)
            with c.builder.if_else(is_none_val) as (then, orelse):
                with then:
                    # Initialize the list value to null otherwise
                    null_ptr = c.context.get_constant_null(t)
                    out_arr_list.inititem(i, null_ptr, incref=False)
                with orelse:
                    n_obj = c.pyapi.call_method(arr_obj, "__len__", ())
                    length = c.pyapi.long_as_longlong(n_obj)
                    c.builder.store(length, len_ptr)
                    c.pyapi.decref(n_obj)
                    arr = c.pyapi.to_native_value(t, arr_obj).value
                    out_arr_list.inititem(i, arr, incref=False)

            c.pyapi.decref(arr_obj)
            c.pyapi.decref(arr_ind_obj)

        setattr(table, f"block_{blk}", out_arr_list.value)

    table.len = c.builder.load(len_ptr)
    c.pyapi.decref(arrs_obj)
    c.pyapi.decref(none_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(table._getvalue(), is_error=is_error)


@box(TableType)
def box_table(typ, val, c, ensure_unboxed=None):
    """Boxes array blocks from native Table into a Python Table"""
    from bodo.hiframes.boxing import get_df_obj_column_codegen

    table = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    # variable number of columns case
    # NOTE: assuming there are no null arrays
    if typ.has_runtime_cols:
        list_arr_typ = types.List(typ.arr_types[0])
        c.context.nrt.incref(c.builder, list_arr_typ, table.block_0)
        table_arr_list_obj = c.pyapi.from_native_value(
            list_arr_typ, table.block_0, c.env_manager
        )
        cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Table))
        out_table_obj = c.pyapi.call_function_objargs(cls_obj, (table_arr_list_obj,))
        c.pyapi.decref(cls_obj)
        c.pyapi.decref(table_arr_list_obj)
        c.context.nrt.decref(c.builder, typ, val)
        return out_table_obj

    table_arr_list_obj = c.pyapi.list_new(
        c.context.get_constant(types.int64, len(typ.arr_types))
    )
    has_parent = cgutils.is_not_null(c.builder, table.parent)
    if ensure_unboxed is None:
        ensure_unboxed = c.context.get_constant(types.bool_, False)

    # generate code for each block
    # box arrays and set into output list object
    for t, blk in typ.type_to_blk.items():
        arr_list = getattr(table, f"block_{blk}")
        arr_list_inst = ListInstance(c.context, c.builder, types.List(t), arr_list)
        # lower array of array indices for block to use within the loop
        # using array since list doesn't have constant lowering
        arr_inds = c.context.make_constant_array(
            c.builder,
            types.Array(types.int64, 1, "C"),
            np.array(typ.block_to_arr_ind[blk], dtype=np.int64),
        )
        arr_inds_struct = c.context.make_array(types.Array(types.int64, 1, "C"))(
            c.context, c.builder, arr_inds
        )
        with cgutils.for_range(c.builder, arr_list_inst.size) as loop:
            i = loop.index
            # get array index in "arrays" list
            arr_ind = _getitem_array_single_int(
                c.context,
                c.builder,
                types.int64,
                types.Array(types.int64, 1, "C"),
                arr_inds_struct,
                i,
            )
            arr = arr_list_inst.getitem(i)
            # set output to None if array value is null
            arr_struct_ptr = cgutils.alloca_once_value(c.builder, arr)
            null_struct_ptr = cgutils.alloca_once_value(
                c.builder, c.context.get_constant_null(t)
            )
            is_null = is_ll_eq(c.builder, arr_struct_ptr, null_struct_ptr)
            with c.builder.if_else(
                c.builder.and_(is_null, c.builder.not_(ensure_unboxed))
            ) as (then, orelse):
                with then:
                    none_obj = c.pyapi.make_none()
                    c.pyapi.list_setitem(table_arr_list_obj, arr_ind, none_obj)
                with orelse:
                    arr_obj = cgutils.alloca_once(
                        c.builder, c.context.get_value_type(types.pyobject)
                    )
                    with c.builder.if_else(c.builder.and_(is_null, has_parent)) as (
                        arr_then,
                        arr_orelse,
                    ):
                        with arr_then:
                            arr_obj_orig = get_df_obj_column_codegen(
                                c.context, c.builder, c.pyapi, table.parent, arr_ind, t
                            )
                            c.builder.store(arr_obj_orig, arr_obj)
                        with arr_orelse:
                            c.context.nrt.incref(c.builder, t, arr)
                            c.builder.store(
                                c.pyapi.from_native_value(t, arr, c.env_manager),
                                arr_obj,
                            )
                    # NOTE: PyList_SetItem() steals a reference so no need to decref
                    # arr_obj
                    c.pyapi.list_setitem(
                        table_arr_list_obj, arr_ind, c.builder.load(arr_obj)
                    )

    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Table))
    out_table_obj = c.pyapi.call_function_objargs(cls_obj, (table_arr_list_obj,))

    c.pyapi.decref(cls_obj)
    c.pyapi.decref(table_arr_list_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return out_table_obj


@overload(len)  # TODO: avoid lowering?
def table_len_overload(T):
    if not isinstance(T, TableType):
        return

    def impl(T):  # pragma: no cover
        return T._len

    return impl


@overload_attribute(TableType, "shape")
def table_shape_overload(T):
    if T.has_runtime_cols:
        # If the number of columns is determined at runtime we can't
        # use compile time values
        def impl(T):  # pragma: no cover
            return (T._len, len(T.block_0))

        return impl

    ncols = len(T.arr_types)
    # using types.int64 due to lowering error (a Numba tuple handling bug)
    return lambda T: (T._len, types.int64(ncols))  # pragma: no cover


def get_table_data_codegen(context, builder, table_arg, col_ind, table_type):
    """generate code for getting a column array from table with original index"""
    arr_type = table_type.arr_types[col_ind]
    table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
    blk = table_type.block_nums[col_ind]
    blk_offset = table_type.block_offsets[col_ind]
    arr_list = getattr(table, f"block_{blk}")
    arr_list_inst = ListInstance(context, builder, types.List(arr_type), arr_list)
    arr = arr_list_inst.getitem(blk_offset)
    return arr


@intrinsic
def get_table_data(typingctx, table_type, ind_typ=None):
    """get data array of table (using the original array index)"""
    assert isinstance(table_type, TableType)
    assert is_overload_constant_int(ind_typ)
    col_ind = get_overload_const_int(ind_typ)
    arr_type = table_type.arr_types[col_ind]

    def codegen(context, builder, sig, args):
        table_arg, _ = args
        arr = get_table_data_codegen(context, builder, table_arg, col_ind, table_type)
        return impl_ret_borrowed(context, builder, arr_type, arr)

    sig = arr_type(table_type, ind_typ)
    return sig, codegen


@intrinsic
def del_column(typingctx, table_type, ind_typ=None):
    """Decrement the reference count by 1 for single column in a table."""
    assert isinstance(table_type, TableType)
    assert is_overload_constant_int(ind_typ)
    col_ind = get_overload_const_int(ind_typ)
    arr_type = table_type.arr_types[col_ind]

    def codegen(context, builder, sig, args):
        table_arg, _ = args
        table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
        # Extract the array from the table
        blk = table_type.block_nums[col_ind]
        blk_offset = table_type.block_offsets[col_ind]
        arr_list = getattr(table, f"block_{blk}")
        arr_list_inst = ListInstance(context, builder, types.List(arr_type), arr_list)
        arr = arr_list_inst.getitem(blk_offset)
        # Decref the array. This decref should ignore nulls, making
        # the operation idempotent.
        context.nrt.decref(builder, arr_type, arr)
        # Set the list value to null to avoid future decref calls
        null_ptr = context.get_constant_null(arr_type)
        arr_list_inst.inititem(blk_offset, null_ptr, incref=False)

    sig = types.void(table_type, ind_typ)
    return sig, codegen


def set_table_data_codegen(
    context,
    builder,
    in_table_type,
    in_table,
    out_table_type,
    arr_type,
    arr_arg,
    col_ind,
    is_new_col,
):
    """generate llvm code for setting array to input table and returning a new table.
    NOTE: this assumes the input table is not used anymore so we can reuse its internal
    lists.
    """
    in_table = cgutils.create_struct_proxy(in_table_type)(context, builder, in_table)
    out_table = cgutils.create_struct_proxy(out_table_type)(context, builder)
    # Copy the length ptr
    out_table.len = in_table.len
    out_table.parent = in_table.parent

    zero = context.get_constant(types.int64, 0)
    one = context.get_constant(types.int64, 1)
    is_new_type = arr_type not in in_table_type.type_to_blk

    # create output blocks
    # avoid list copy overhead since modifying input is ok in all cases
    # NOTE: we may also increase the list size for an input block which is ok
    # copy blocks from input table for other arrays
    for t, blk in out_table_type.type_to_blk.items():
        if t in in_table_type.type_to_blk:
            in_blk = in_table_type.type_to_blk[t]
            out_arr_list = ListInstance(
                context,
                builder,
                types.List(t),
                getattr(in_table, f"block_{in_blk}"),
            )
            context.nrt.incref(builder, types.List(t), out_arr_list.value)
            setattr(out_table, f"block_{blk}", out_arr_list.value)

    # 5 cases, new array is:
    # 1) new column, new type (create new block)
    # 2) new column, existing type (append to existing block)
    # 3) existing column, new type (create new block, remove previous array)
    # 4) existing column, existing type, same type as before (replace array)
    # 5) existing column, existing type, different type than before
    # (remove previous array, insert new array)

    # new type cases (1, 3)
    if is_new_type:
        # create a new list for new type
        _, out_arr_list = ListInstance.allocate_ex(
            context, builder, types.List(arr_type), one
        )
        out_arr_list.size = one
        out_arr_list.inititem(zero, arr_arg, incref=True)
        blk = out_table_type.type_to_blk[arr_type]
        setattr(out_table, f"block_{blk}", out_arr_list.value)

        # case 3: if replacing an existing column, the old array value has to be removed
        if not is_new_col:
            _rm_old_array(
                col_ind,
                out_table_type,
                out_table,
                in_table_type,
                context,
                builder,
            )

    # existing type cases (2, 4, 5)
    else:
        blk = out_table_type.type_to_blk[arr_type]
        out_arr_list = ListInstance(
            context,
            builder,
            types.List(arr_type),
            getattr(out_table, f"block_{blk}"),
        )
        # case 2: append at end of list if new column
        if is_new_col:
            n = out_arr_list.size
            new_size = builder.add(n, one)
            out_arr_list.resize(new_size)
            out_arr_list.inititem(n, arr_arg, incref=True)
        # case 4: not new column, replace existing value
        elif arr_type == in_table_type.arr_types[col_ind]:
            # input/output offsets should be the same if not new column
            offset = context.get_constant(
                types.int64, out_table_type.block_offsets[col_ind]
            )
            out_arr_list.setitem(offset, arr_arg, True)
        # case 5: remove old array value, insert new array value
        else:
            _rm_old_array(
                col_ind,
                out_table_type,
                out_table,
                in_table_type,
                context,
                builder,
            )
            offset = context.get_constant(
                types.int64, out_table_type.block_offsets[col_ind]
            )
            # similar to list.insert() code in Numba:
            # https://github.com/numba/numba/blob/805e24fbd895d90634cca68c13f4c439609e9286/numba/cpython/listobj.py#L977
            n = out_arr_list.size
            new_size = builder.add(n, one)
            out_arr_list.resize(new_size)
            # need to add an extra incref since setitem decrefs existing value
            # https://github.com/numba/numba/issues/7553
            context.nrt.incref(builder, arr_type, out_arr_list.getitem(offset))
            out_arr_list.move(builder.add(offset, one), offset, builder.sub(n, offset))
            out_arr_list.setitem(offset, arr_arg, incref=True)

    return out_table._getvalue()


def _rm_old_array(col_ind, out_table_type, out_table, in_table_type, context, builder):
    """helper function for set_table_data_codegen() to remove array value from block"""
    old_type = in_table_type.arr_types[col_ind]
    # corner case: the old type had only one array which is removed in
    # output table already (there is no type block for old_type anymore)
    if old_type in out_table_type.type_to_blk:
        blk = out_table_type.type_to_blk[old_type]
        old_type_list = getattr(out_table, f"block_{blk}")
        lst_type = types.List(old_type)
        # using offset from in_table_type since out_table_type doesn't
        # include this array
        offset = context.get_constant(types.int64, in_table_type.block_offsets[col_ind])
        # array_list.pop(offset)
        pop_sig = lst_type.dtype(lst_type, types.intp)
        old_arr = context.compile_internal(
            builder,
            lambda lst, i: lst.pop(i),
            pop_sig,
            (old_type_list, offset),
        )  # pragma: no cover
        context.nrt.decref(builder, old_type, old_arr)


@intrinsic
def set_table_data(typingctx, table_type, ind_type, arr_type=None):
    """Set array to input table and return a new table.
    NOTE: this assumes the input table is not used anymore so we can reuse its internal
    lists.
    """
    assert isinstance(table_type, TableType), "invalid input to set_table_data"
    assert is_overload_constant_int(ind_type), "set_table_data expects const index"
    col_ind = get_overload_const_int(ind_type)
    is_new_col = col_ind == len(table_type.arr_types)
    out_arr_types = list(table_type.arr_types)
    if is_new_col:
        out_arr_types.append(arr_type)
    else:
        out_arr_types[col_ind] = arr_type
    out_table_type = TableType(tuple(out_arr_types))

    def codegen(context, builder, sig, args):
        table_arg, _, new_arr = args
        out_table = set_table_data_codegen(
            context,
            builder,
            table_type,
            table_arg,
            out_table_type,
            arr_type,
            new_arr,
            col_ind,
            is_new_col,
        )
        return out_table

    return out_table_type(table_type, ind_type, arr_type), codegen


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("get_table_data", "bodo.hiframes.table")
] = alias_ext_dummy_func


def get_table_data_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for get_table_data(). output array has the same length as input
    table.
    """
    assert len(args) == 2 and not kws
    var = args[0]

    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(var)[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_table_get_table_data = get_table_data_equiv

# TODO: ArrayAnalysis for set_table_data?


@lower_constant(TableType)
def lower_constant_table(context, builder, table_type, pyval):
    """embed constant Table value by getting constant values for data arrays."""

    table = cgutils.create_struct_proxy(table_type)(context, builder)
    table.parent = cgutils.get_null_value(table.parent.type)

    # create each array type block
    for t, blk in table_type.type_to_blk.items():
        blk_n_arrs = len(table_type.block_to_arr_ind[blk])
        n_arrs = context.get_constant(types.int64, blk_n_arrs)
        _, out_arr_list = ListInstance.allocate_ex(
            context, builder, types.List(t), n_arrs
        )
        out_arr_list.size = n_arrs

        for i in range(blk_n_arrs):
            arr_ind = table_type.block_to_arr_ind[blk][i]
            arr = context.get_constant_generic(
                builder, table_type.arr_types[arr_ind], pyval.arrays[arr_ind]
            )
            list_ind = context.get_constant(types.int64, i)
            out_arr_list.inititem(list_ind, arr, incref=False)

        setattr(table, f"block_{blk}", out_arr_list.value)

    return table._getvalue()


@intrinsic
def init_table(typingctx, table_type=None):
    """initialize a table object with same structure as input table without setting it's
    array blocks (to be set later)
    """
    assert isinstance(table_type, TableType), "table type expected"

    def codegen(context, builder, sig, args):
        table = cgutils.create_struct_proxy(table_type)(context, builder)
        for t, blk in table_type.type_to_blk.items():
            null_list = context.get_constant_null(types.List(t))
            setattr(table, f"block_{blk}", null_list)
        return table._getvalue()

    sig = table_type(table_type)
    return sig, codegen


@intrinsic
def get_table_block(typingctx, table_type, blk_type=None):
    """get array list for a type block of table"""
    assert isinstance(table_type, TableType), "table type expected"
    assert is_overload_constant_int(blk_type)
    blk = get_overload_const_int(blk_type)
    arr_type = None
    for t, b in table_type.type_to_blk.items():
        if b == blk:
            arr_type = t
            break
    assert arr_type is not None, "invalid table type block"
    out_list_type = types.List(arr_type)

    def codegen(context, builder, sig, args):
        table = cgutils.create_struct_proxy(table_type)(context, builder, args[0])
        arr_list = getattr(table, f"block_{blk}")
        return impl_ret_borrowed(context, builder, out_list_type, arr_list)

    sig = out_list_type(table_type, blk_type)
    return sig, codegen


@intrinsic
def ensure_column_unboxed(typingctx, table_type, arr_list_t, ind_t, arr_ind_t=None):
    """make sure column of table is unboxed
    Throw an error if column array is null and there is no parent to unbox from
    """
    assert isinstance(table_type, TableType), "table type expected"
    sig = types.none(table_type, arr_list_t, ind_t, arr_ind_t)
    return sig, ensure_column_unboxed_codegen


def ensure_column_unboxed_codegen(context, builder, sig, args):
    """
    Codegen for ensure_column_unboxed. This isn't a closure so it can be
    used by intrinsics.
    """
    from bodo.hiframes.boxing import get_df_obj_column_codegen

    table_arg, list_arg, i_arg, arr_ind_arg = args
    pyapi = context.get_python_api(builder)

    table = cgutils.create_struct_proxy(sig.args[0])(context, builder, table_arg)
    has_parent = cgutils.is_not_null(builder, table.parent)

    arr_list_inst = ListInstance(context, builder, sig.args[1], list_arg)
    in_arr = arr_list_inst.getitem(i_arg)

    arr_struct_ptr = cgutils.alloca_once_value(builder, in_arr)
    null_struct_ptr = cgutils.alloca_once_value(
        builder, context.get_constant_null(sig.args[1].dtype)
    )
    is_null = is_ll_eq(builder, arr_struct_ptr, null_struct_ptr)
    with builder.if_then(is_null):
        with builder.if_else(has_parent) as (then, orelse):
            with then:
                arr_obj = get_df_obj_column_codegen(
                    context,
                    builder,
                    pyapi,
                    table.parent,
                    arr_ind_arg,
                    sig.args[1].dtype,
                )
                arr = pyapi.to_native_value(sig.args[1].dtype, arr_obj).value
                arr_list_inst.inititem(i_arg, arr, incref=False)
                pyapi.decref(arr_obj)
            with orelse:
                context.call_conv.return_user_exc(
                    builder, BodoError, ("unexpected null table column",)
                )


@intrinsic
def set_table_block(typingctx, table_type, arr_list_type, blk_type=None):
    """set table block and return a new table object"""
    assert isinstance(table_type, TableType), "table type expected"
    assert isinstance(arr_list_type, types.List), "list type expected"
    assert is_overload_constant_int(blk_type), "blk should be const int"
    blk = get_overload_const_int(blk_type)

    def codegen(context, builder, sig, args):
        table_arg, arr_list_arg, _ = args
        in_table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
        setattr(in_table, f"block_{blk}", arr_list_arg)
        return impl_ret_borrowed(context, builder, table_type, in_table._getvalue())

    sig = table_type(table_type, arr_list_type, blk_type)
    return sig, codegen


@intrinsic
def set_table_len(typingctx, table_type, l_type=None):
    """set table len and return a new table object"""
    assert isinstance(table_type, TableType), "table type expected"

    def codegen(context, builder, sig, args):
        table_arg, l_arg = args
        in_table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
        in_table.len = l_arg
        return impl_ret_borrowed(context, builder, table_type, in_table._getvalue())

    sig = table_type(table_type, l_type)
    return sig, codegen


@intrinsic
def alloc_list_like(typingctx, list_type=None):
    """
    allocate a list with same type and size as input list but filled with null values
    """
    assert isinstance(list_type, types.List), "list type expected"

    def codegen(context, builder, sig, args):
        in_list = ListInstance(context, builder, list_type, args[0])
        size = in_list.size
        _, out_arr_list = ListInstance.allocate_ex(context, builder, list_type, size)
        out_arr_list.size = size
        return out_arr_list.value

    sig = list_type(list_type)
    return sig, codegen


def _get_idx_length(idx):  # pragma: no cover
    pass


@overload(_get_idx_length)
def overload_get_idx_length(idx, n):
    """get length of boolean array or slice index"""
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        return lambda idx, n: idx.sum()  # pragma: no cover

    assert isinstance(idx, types.SliceType), "slice index expected"

    def impl(idx, n):  # pragma: no cover
        slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
        return numba.cpython.unicode._slice_span(slice_idx)

    return impl


def gen_table_filter(T, used_cols=None):
    """Generate a function that filters table using input boolean array or slice.
    If used_cols is passed, only used columns are written in output.
    """
    from bodo.utils.conversion import ensure_contig_if_np

    glbls = {
        "init_table": init_table,
        "get_table_block": get_table_block,
        "ensure_column_unboxed": ensure_column_unboxed,
        "set_table_block": set_table_block,
        "set_table_len": set_table_len,
        "alloc_list_like": alloc_list_like,
        "_get_idx_length": _get_idx_length,
        "ensure_contig_if_np": ensure_contig_if_np,
    }
    if used_cols is not None:
        glbls["used_cols"] = np.array(used_cols, dtype=np.int64)

    func_text = "def impl(T, idx):\n"
    func_text += f"  T2 = init_table(T)\n"
    func_text += f"  l = 0\n"

    # set table length using index value and return if no table column is used
    if used_cols is not None and len(used_cols) == 0:
        # avoiding _get_idx_length in the general case below since it has extra overhead
        func_text += f"  l = _get_idx_length(idx, len(T))\n"
        func_text += f"  T2 = set_table_len(T2, l)\n"
        func_text += f"  return T2\n"
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        return loc_vars["impl"]

    if used_cols is not None:
        func_text += f"  used_set = set(used_cols)\n"
    for blk in T.type_to_blk.values():
        glbls[f"arr_inds_{blk}"] = np.array(T.block_to_arr_ind[blk], dtype=np.int64)
        func_text += f"  arr_list_{blk} = get_table_block(T, {blk})\n"
        func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_{blk})\n"
        func_text += f"  for i in range(len(arr_list_{blk})):\n"
        func_text += f"    arr_ind_{blk} = arr_inds_{blk}[i]\n"
        if used_cols is not None:
            func_text += f"    if arr_ind_{blk} not in used_set: continue\n"
        func_text += f"    ensure_column_unboxed(T, arr_list_{blk}, i, arr_ind_{blk})\n"
        func_text += (
            f"    out_arr_{blk} = ensure_contig_if_np(arr_list_{blk}[i][idx])\n"
        )
        func_text += f"    l = len(out_arr_{blk})\n"
        func_text += f"    out_arr_list_{blk}[i] = out_arr_{blk}\n"
        func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
    func_text += f"  T2 = set_table_len(T2, l)\n"
    func_text += f"  return T2\n"

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl"]


@overload(operator.getitem, no_unliteral=True)
def table_getitem(T, idx):
    if not isinstance(T, TableType):
        return

    return gen_table_filter(T)


@intrinsic
def init_runtime_table_from_list(typingctx, arr_list_typ, nrows_typ=None):
    """
    Takes a list of arrays and the length of each array and creates a Python
    Table from this list (without making a copy).

    The number of arrays in this table is NOT known at compile time.
    """
    assert isinstance(
        arr_list_typ, types.List
    ), "init_runtime_table_from_list requires a list of arrays"
    assert isinstance(
        nrows_typ, types.Integer
    ), "init_runtime_table_from_list requires an integer length"

    def codegen(context, builder, sig, args):
        (arr_list, nrows) = args
        table = cgutils.create_struct_proxy(table_type)(context, builder)
        # Update the table length. This is assume to be 0 if the arr_list is empty
        table.len = nrows
        # Store the array list in the table and increment its refcount.
        table.block_0 = arr_list
        context.nrt.incref(builder, arr_list_typ, arr_list)
        return table._getvalue()

    table_type = TableType((arr_list_typ.dtype,), True)
    sig = table_type(arr_list_typ, nrows_typ)
    return sig, codegen
