"""Table data type for storing dataframe column arrays. Supports storing many columns
(e.g. >10k) efficiently.
"""

import operator
from collections import defaultdict
from functools import cached_property

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, ir, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.core.ir_utils import guard
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.cpython.listobj import ListInstance
from numba.extending import (
    NativeValue,
    box,
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_getattr,
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    typeof_impl,
    unbox,
)
from numba.np.arrayobj import _getitem_array_single_int
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.utils.cg_helpers import is_ll_eq
from bodo.utils.templates import OverloadedKeyAttributeTemplate
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    MetaType,
    assert_bodo_error,
    decode_if_dict_array,
    get_overload_const_int,
    is_list_like_index_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_none,
    is_overload_true,
    raise_bodo_error,
    to_str_arr_if_dict_array,
    unwrap_typeref,
)
from bodo.utils.utils import (
    alloc_type,
    bodo_exec,
    cached_call_internal,
    is_array_typ,
    is_whole_slice,
    numba_to_c_array_types,
    numba_to_c_types,
    set_wrapper,
)


class Table:
    """basic table which is just a list of arrays
    Python definition needed since CSV reader passes arrays from objmode
    """

    def __init__(self, arrs, usecols=None, num_arrs=-1, dist=None):
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
        from bodo.transforms.distributed_analysis import Distribution

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
        self.dist = Distribution.REP.value if dist is None else dist

    def __eq__(self, other):
        return (
            isinstance(other, Table)
            and len(self.arrays) == len(other.arrays)
            and all((a == b).all() for a, b in zip(self.arrays, other.arrays))
        )

    def __len__(self):
        return len(self.arrays[0]) if len(self.arrays) > 0 else 0

    def __str__(self) -> str:
        return f"Table({str(self.arrays)})"

    def __repr__(self) -> str:
        return f"Table({repr(self.arrays)})"

    def __getitem__(self, val):
        if not isinstance(val, slice):
            raise TypeError("Table getitem only supported for slices")
        # Just slice each array
        return Table(
            [arr if arr is None else arr[val] for arr in self.arrays], dist=self.dist
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.arrays[0] if len(self.arrays) > 0 else 0), len(self.arrays))

    def to_pandas(self, index=None):
        """convert table to a DataFrame (with column names just a range of numbers)"""
        n_cols = len(self.arrays)
        data = dict(zip(range(n_cols), self.arrays))
        df = pd.DataFrame(data, index)
        return df


def _unify_array_types(typingctx, t1, t2):
    """Unify two array types for table unification.
    Uses _derive_common_key_type() if necessary to expand integer types since regular
    unify functions don't handle that case (necessary for runtime join filter typing,
    and doesn't hurt other use cases).
    """
    if t1 == t2:
        return t1

    unified = typingctx.unify_types(t1, t2)

    if unified is not None:
        return unified

    # TODO[BSE-4462]: revisit table unification functions

    # NOTE: may raise BodoError if cannot unify
    return bodo.libs.streaming.join.JoinStateType._derive_common_key_type([t1, t2])


class TableType(types.ArrayCompatible):
    """Bodo Table type that stores column arrays for DataFrames.
    Arrays of the same type are stored in the same "block" (kind of similar to Pandas).
    This allows for loop generation for columns of same type instead of generating code
    for each column (important for DataFrames with many columns).
    """

    def __init__(
        self,
        arr_types: tuple[types.ArrayCompatible, ...],
        has_runtime_cols: bool = False,
        dist=None,
    ):
        from bodo.transforms.distributed_analysis import Distribution

        self.arr_types = arr_types
        self.has_runtime_cols = has_runtime_cols

        # block number for each array in arr_types
        block_nums = []
        # offset within block for each array in arr_types
        block_offsets = []
        # block number for each array type
        type_to_blk = {}
        # array type to block number.
        # reverse of type_to_blk
        blk_to_type = {}
        # current number of arrays in the block
        blk_curr_ind = defaultdict(int)
        # indices of arrays in arr_types for each block
        block_to_arr_ind = defaultdict(list)
        # We only don't have mapping information if
        # columns are only known at runtime.
        if not has_runtime_cols:
            for i, t in enumerate(arr_types):
                if t not in type_to_blk:
                    next_blk = len(type_to_blk)
                    type_to_blk[t] = next_blk
                    blk_to_type[next_blk] = t

                blk = type_to_blk[t]
                block_nums.append(blk)
                block_offsets.append(blk_curr_ind[blk])
                blk_curr_ind[blk] += 1
                block_to_arr_ind[blk].append(i)

        self.block_nums = block_nums
        self.block_offsets = block_offsets
        self.type_to_blk = type_to_blk
        self.blk_to_type = blk_to_type
        self.block_to_arr_ind = block_to_arr_ind

        dist = Distribution.OneD_Var if dist is None else dist
        self.dist = dist
        super().__init__(name=f"TableType({arr_types}, {has_runtime_cols}, {dist})")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 2, "C")

    @property
    def key(self):
        return self.arr_types, self.has_runtime_cols

    def unify(self, typingctx, other):
        """Unify two TableType instances (required for runtime join filter typing)."""
        if (
            isinstance(other, TableType)
            and (len(self.arr_types) == len(other.arr_types))
            # TODO: revisit unify for runtime columns case
            and (not self.has_runtime_cols and not other.has_runtime_cols)
            and (self.dist == other.dist)
        ):
            try:
                new_arr_types = tuple(
                    _unify_array_types(typingctx, t1, t2)
                    for t1, t2 in zip(self.arr_types, other.arr_types)
                )
            except BodoError:
                return None
            return TableType(new_arr_types, self.has_runtime_cols, self.dist)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    def copy(self, dist=None):
        if dist is None:
            dist = self.dist
        return TableType(self.arr_types, self.has_runtime_cols, dist)

    @cached_property
    def c_array_types(self) -> list[int]:
        return numba_to_c_array_types(self.arr_types)

    @cached_property
    def c_dtypes(self) -> list[int]:
        return numba_to_c_types(self.arr_types)


@typeof_impl.register(Table)
def typeof_table(val, c):
    from bodo.transforms.distributed_analysis import Distribution

    return TableType(
        tuple(numba.typeof(arr) for arr in val.arrays), dist=Distribution(val.dist)
    )


@register_model(TableType)
class TableTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # store a list of arrays for each block of same type column arrays
        if fe_type.has_runtime_cols:
            members = [
                (f"block_{i}", types.List(t)) for i, t in enumerate(fe_type.arr_types)
            ]
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
        super().__init__(dmm, fe_type, members)


# for debugging purposes (a table may not have a block)
make_attribute_wrapper(TableType, "block_0", "block_0")
make_attribute_wrapper(TableType, "len", "_len")


@infer_getattr
class TableTypeAttribute(OverloadedKeyAttributeTemplate):
    """
    Attribute template for Table. This is used to
    avoid lowering operations whose typing can have
    a large impact on compilation.
    """

    key = TableType

    def resolve_shape(self, df):
        return types.Tuple([types.int64, types.int64])


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
        # Compute the size of the output list
        list_size = c.context.get_constant(types.int64, 0)
        for i, t in enumerate(typ.arr_types):
            arr_list = getattr(table, f"block_{i}")
            arr_list_inst = ListInstance(c.context, c.builder, types.List(t), arr_list)
            list_size = c.builder.add(list_size, arr_list_inst.size)
        table_arr_list_obj = c.pyapi.list_new(list_size)
        # Store the list elements
        curr_idx = c.context.get_constant(types.int64, 0)
        for i, t in enumerate(typ.arr_types):
            arr_list = getattr(table, f"block_{i}")
            arr_list_inst = ListInstance(c.context, c.builder, types.List(t), arr_list)
            with cgutils.for_range(c.builder, arr_list_inst.size) as loop:
                i = loop.index
                arr = arr_list_inst.getitem(i)
                c.context.nrt.incref(c.builder, t, arr)
                idx = c.builder.add(curr_idx, i)
                c.pyapi.list_setitem(
                    table_arr_list_obj,
                    idx,
                    c.pyapi.from_native_value(t, arr, c.env_manager),
                )
            curr_idx = c.builder.add(curr_idx, arr_list_inst.size)

        # Compute the output table
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

    dist_val_obj = c.pyapi.long_from_longlong(
        lir.Constant(lir.IntType(64), typ.dist.value)
    )
    c.pyapi.object_setattr_string(out_table_obj, "dist", dist_val_obj)

    c.pyapi.decref(cls_obj)
    c.pyapi.decref(table_arr_list_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return out_table_obj


@lower_builtin(len, TableType)
def table_len_lower(context, builder, sig, args):
    """
    Implementation for lowering len. The typing is
    done in a shared template for many different types.
    See LenTemplate.
    """
    return context.compile_internal(builder, lambda T: T._len, sig, args)


def local_len(x):
    """
    Determine the rank-local length of an object.
    Right now, only implemented for tables, but should be expanded
    to other datatypes in the future
    """


# We can reuse the same overload internally
# Since distributed pass rewrites __builtin__.len() only
@infer_global(local_len)
class LocalLenInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert len(args) == 1 and not kws
        if not isinstance(args[0], TableType):
            raise BodoError("local_len() only supported for tables")
        return signature(types.int64, args[0])


@lower_builtin(local_len, TableType)
def local_len_lower(context, builder, sig, args):
    return context.compile_internal(builder, lambda T: T._len, sig, args)


@lower_getattr(TableType, "shape")
def lower_table_shape(context, builder, typ, val):
    """
    Lowering for TableType.shape. This compile and calls
    an implementation with overload style.
    """
    impl = table_shape_overload(typ)
    return cached_call_internal(
        context, builder, impl, types.Tuple([types.int64, types.int64])(typ), (val,)
    )


def table_shape_overload(T):
    """
    Actual implementation used to implement TableType.shape.
    """
    if T.has_runtime_cols:
        # If the number of columns is determined at runtime we can't
        # use compile time values
        def impl(T):  # pragma: no cover
            return (T._len, compute_num_runtime_columns(T))

        return impl

    ncols = len(T.arr_types)
    # using types.int64 due to lowering error (a Numba tuple handling bug)
    return lambda T: (T._len, types.int64(ncols))  # pragma: no cover


@intrinsic
def compute_num_runtime_columns(typingctx, table_type):
    """
    Compute the number of columns generated for a table
    with columns at runtime.
    """
    assert isinstance(table_type, TableType)

    def codegen(context, builder, sig, args):
        (table_arg,) = args
        table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
        num_cols = context.get_constant(types.int64, 0)
        for i, t in enumerate(table_type.arr_types):
            arr_list = getattr(table, f"block_{i}")
            arr_list_inst = ListInstance(context, builder, types.List(t), arr_list)
            num_cols = builder.add(num_cols, arr_list_inst.size)
        return num_cols

    sig = types.int64(table_type)
    return sig, codegen


def get_table_data_codegen(context, builder, table_arg, col_ind, table_type):
    """generate code for getting a column array from table with original index"""
    arr_type = table_type.arr_types[col_ind]
    table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
    blk = table_type.block_nums[col_ind]
    blk_offset = table_type.block_offsets[col_ind]
    arr_list = getattr(table, f"block_{blk}")

    # unbox the column if necessary
    unbox_sig = types.none(table_type, types.List(arr_type), types.int64, types.int64)
    col_ind_arg = context.get_constant(types.int64, col_ind)
    blk_offset_arg = context.get_constant(types.int64, blk_offset)
    unbox_args = (table_arg, arr_list, blk_offset_arg, col_ind_arg)
    ensure_column_unboxed_codegen(context, builder, unbox_sig, unbox_args)

    arr_list_inst = ListInstance(context, builder, types.List(arr_type), arr_list)
    arr = arr_list_inst.getitem(blk_offset)
    return arr


@intrinsic(prefer_literal=True)
def get_table_data(typingctx, table_type, ind_typ):
    """get data array of table (using the original array index)"""
    assert isinstance(table_type, TableType)
    assert_bodo_error(is_overload_constant_int(ind_typ))
    col_ind = get_overload_const_int(ind_typ)
    arr_type = table_type.arr_types[col_ind]

    def codegen(context, builder, sig, args):
        table_arg, _ = args
        arr = get_table_data_codegen(context, builder, table_arg, col_ind, table_type)
        return impl_ret_borrowed(context, builder, arr_type, arr)

    sig = arr_type(table_type, ind_typ)
    return sig, codegen


@intrinsic
def del_column(typingctx, table_type, ind_typ):
    """Decrement the reference count by 1 for the columns in a table."""
    from bodo.io.arrow_reader import ArrowReaderType

    # Right now in TableColumnDelPass, we treat ArrowReaderType as a TableType
    # in order to perform column pruning for streaming IO. However, when the
    # ArrowReaderType is used outside of the streaming loop (to call C++
    # delete, for example), TableColumnDelPass adds del_column calls for dead
    # columns after the loop. Current hacky solution is to have del_column be
    # a no-op in this case
    # TODO: Properly track and handle operator objects in TableColumnDelPass
    if isinstance(table_type, ArrowReaderType):  # pragma: no cover
        sig = types.void(table_type, ind_typ)

        def codegen(context, builder, sig, args):
            return

        return sig, codegen

    assert isinstance(table_type, TableType), "Can only delete columns from a table"
    assert isinstance(ind_typ, types.TypeRef) and isinstance(
        ind_typ.instance_type, MetaType
    ), "ind_typ must be a typeref for a meta type"
    col_inds = list(ind_typ.instance_type.meta)
    # Determine which blocks and column numbers need decrefs.
    block_del_inds = defaultdict(list)
    for ind in col_inds:
        block_del_inds[table_type.block_nums[ind]].append(table_type.block_offsets[ind])

    def codegen(context, builder, sig, args):
        table_arg, _ = args
        table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
        for blk, blk_offsets in block_del_inds.items():
            arr_type = table_type.blk_to_type[blk]
            arr_list = getattr(table, f"block_{blk}")
            arr_list_inst = ListInstance(
                context, builder, types.List(arr_type), arr_list
            )
            # Create a single null array for this type
            null_ptr = context.get_constant_null(arr_type)
            if len(blk_offsets) == 1:
                # get_dataframe_data will select individual
                # columns, so we need to generate more efficient
                # code for the single column case.
                # TODO: Determine a reasonable threshold?
                blk_offset = blk_offsets[0]
                # Extract the array from the table
                arr = arr_list_inst.getitem(blk_offset)
                # Decref the array. This decref should ignore nulls, making
                # the operation idempotent.
                context.nrt.decref(builder, arr_type, arr)
                # Set the list value to null to avoid future decref calls
                arr_list_inst.inititem(blk_offset, null_ptr, incref=False)
            else:
                # Generate a for loop for several decrefs in the same
                # block.
                n_arrs = context.get_constant(types.int64, len(blk_offsets))
                # lower array of block offsets to use in the loop
                blk_offset_arr = context.make_constant_array(
                    builder,
                    types.Array(types.int64, 1, "C"),
                    # On windows np.array defaults to the np.int32 for integers.
                    # As a result, we manually specify int64 during the array
                    # creation to keep the lowered constant consistent with the
                    # expected type.
                    np.array(blk_offsets, dtype=np.int64),
                )
                blk_offset_arr_struct = context.make_array(
                    types.Array(types.int64, 1, "C")
                )(context, builder, blk_offset_arr)
                with cgutils.for_range(builder, n_arrs) as loop:
                    i = loop.index
                    # get array index in "arrays"
                    blk_offset = _getitem_array_single_int(
                        context,
                        builder,
                        types.int64,
                        types.Array(types.int64, 1, "C"),
                        blk_offset_arr_struct,
                        i,
                    )
                    # Extract the array from the table
                    arr = arr_list_inst.getitem(blk_offset)
                    context.nrt.decref(builder, arr_type, arr)
                    # Set the list value to null to avoid future decref calls
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
            out_arr_list.setitem(offset, arr_arg, incref=True)
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


def generate_set_table_data_code(table, ind, arr_type, used_cols, is_null=False):
    """Generates the code used for both set_table_data and
    set_table_data_null, which are distinguished by the null
    parameter. Note: Since we copy the parent, we do not need to
    ensure any columns are unboxed as no data is actually manipulated.

    Args:
        table (TableType): Input table type that will have a column set
        ind (int): Index at which the new array will be placed.
        arr_type (ArrayType): Type of the new array being set. Used to
            generate code/determine the output table type.
        used_cols ([Set[int] | None]: If not None, the columns that should
            be copied to the output table. This is filled by the table column
            deletion steps. Since set_table_data, is a simple assignment, we only
            use this set to skip entire loops and do not do a runtime check if
            a loop is used. Note: used_cols never contains ind.
        is_null (bool, optional): If this set_table_data_null? If so
            the arr in the codegen is actually just a type, not an
            actual array and we don't set the value within the output
            table.

    Returns:
        func: Python func with the generated code for either set_table_data
        or set_table_data_null
    """
    out_arr_typs = list(table.arr_types)
    if ind == len(out_arr_typs):
        old_arr_type = None
        out_arr_typs.append(arr_type)
    else:
        old_arr_type = table.arr_types[ind]
        out_arr_typs[ind] = arr_type
    out_table_typ = TableType(tuple(out_arr_typs))
    glbls = {
        "init_table": init_table,
        "get_table_block": get_table_block,
        "set_table_block": set_table_block,
        "set_table_len": set_table_len,
        "set_table_parent": set_table_parent,
        "alloc_list_like": alloc_list_like,
        "out_table_typ": out_table_typ,
    }
    func_text = "def bodo_set_table_data(table, ind, arr, used_cols=None):\n"
    func_text += "  T2 = init_table(out_table_typ, False)\n"
    # Length of the table cannot change.
    func_text += "  T2 = set_table_len(T2, len(table))\n"
    # Copy the parent for lazy unboxing.
    func_text += "  T2 = set_table_parent(T2, table)\n"
    for typ, blk in out_table_typ.type_to_blk.items():
        if typ in table.type_to_blk:
            orig_table_blk = table.type_to_blk[typ]
            func_text += (
                f"  arr_list_{blk} = get_table_block(table, {orig_table_blk})\n"
            )
            func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_{blk}, {len(out_table_typ.block_to_arr_ind[blk])}, False)\n"
            # If this typ isn't included skip the whole loop.
            if used_cols is None or (
                set(table.block_to_arr_ind[orig_table_blk]) & used_cols
            ):
                func_text += f"  for i in range(len(arr_list_{blk})):\n"
                if typ not in (old_arr_type, arr_type):
                    func_text += f"    out_arr_list_{blk}[i] = arr_list_{blk}[i]\n"
                else:
                    # If the contents of the list has changed then we need to actually lower
                    # the location for each column.
                    ind_list = table.block_to_arr_ind[orig_table_blk]
                    blk_idx_lst = np.empty(len(ind_list), np.int64)
                    removes_arr = False
                    # Arrays are always added to the list in logical column order.
                    for blk_idx, arr_ind in enumerate(ind_list):
                        if arr_ind != ind:
                            new_blk_idx = out_table_typ.block_offsets[arr_ind]
                        else:
                            new_blk_idx = -1
                            removes_arr = True
                        blk_idx_lst[blk_idx] = new_blk_idx

                    glbls[f"out_idxs_{blk}"] = np.array(blk_idx_lst, np.int64)
                    func_text += f"    out_idx = out_idxs_{blk}[i]\n"
                    if removes_arr:
                        # If we remove any array we need to check for -1
                        func_text += "    if out_idx == -1:\n"
                        func_text += "      continue\n"
                    func_text += (
                        f"    out_arr_list_{blk}[out_idx] = arr_list_{blk}[i]\n"
                    )
            if typ == arr_type and not is_null:
                # Add the new array.
                func_text += (
                    f"  out_arr_list_{blk}[{out_table_typ.block_offsets[ind]}] = arr\n"
                )
        else:
            # We are assigning a new type.
            glbls[f"arr_list_typ_{blk}"] = types.List(arr_type)
            func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_typ_{blk}, 1, False)\n"
            if not is_null:
                # Assign the array if it exists
                func_text += f"  out_arr_list_{blk}[0] = arr\n"
        func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
    func_text += "  return T2\n"

    return bodo_exec(func_text, glbls, {}, __name__)


@numba.generated_jit(
    nopython=True, no_cpython_wrapper=True, no_unliteral=True, cache=True
)
def set_table_data(table, ind, arr, used_cols=None):
    """Returns a new table with the contents as table
    except arr is inserted at ind. There are two main cases,
    if ind = len(table.arr_types) then we are adding a new array.
    Otherwise ind must be in range(0, len(table.arr_types)) and
    we replace the existing array.

    Args:
        table (TableType): Input table.
        ind (LiteralInteger): Location at which to place the new array.
        arr (ArrayType): The new array to place
        used_cols (types.TypeRef(bodo.MetaType)
            | types.none): If not None, the columns that should
            be copied to the output table. This is filled by the table
            column deletion steps.

    Returns:
        TableType: Output table with the new array.
    """
    if is_overload_none(used_cols):
        used_columns = None
    else:
        used_columns = set(used_cols.instance_type.meta)
    ind_lit = get_overload_const_int(ind)
    return generate_set_table_data_code(table, ind_lit, arr, used_columns)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True, no_unliteral=True)
def set_table_data_null(table, ind, arr, used_cols=None):
    """Returns a new table with the contents as table
    except a new null array is inserted at ind. There are two main cases,
    if ind = len(table.arr_types) then we are just modifying the table
    types with modifying the data. Otherwise ind must be in
    range(0, len(table.arr_types)) and we replace the existing array
    with null.

    Args:
        table (TableType): Input table.
        ind (LiteralInteger): Location at which to place the new array.
        arr (TypeRef(ArrayType)): The type of the new null array to place.
            This is used for generating an accurate table type.
        used_cols (types.TypeRef(bodo.MetaType)
            | types.none): If not None, the columns that should
            be copied to the output table. This is filled by the table
            column deletion steps.

    Returns:
        TableType: Output table with the new null array.
    """
    ind_lit = get_overload_const_int(ind)
    arr_type = arr.instance_type
    if is_overload_none(used_cols):
        used_columns = None
    else:
        used_columns = set(used_cols.instance_type.meta)
    return generate_set_table_data_code(
        table, ind_lit, arr_type, used_columns, is_null=True
    )


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[("get_table_data", "bodo.hiframes.table")] = (
    alias_ext_dummy_func
)


def get_table_data_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for get_table_data(). output array has the same length as input
    table.
    """
    assert len(args) == 2 and not kws
    var = args[0]

    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(var)[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_table_get_table_data = get_table_data_equiv

# TODO: ArrayAnalysis for set_table_data and set_table_data_null?


@lower_constant(TableType)
def lower_constant_table(context, builder, table_type, pyval):
    """embed constant Table value by getting constant values for data arrays."""

    arr_lists = []
    # create each array type block
    for t, blk in table_type.type_to_blk.items():
        blk_n_arrs = len(table_type.block_to_arr_ind[blk])
        t_arr_list = []
        for i in range(blk_n_arrs):
            arr_ind = table_type.block_to_arr_ind[blk][i]
            t_arr_list.append(pyval.arrays[arr_ind])

        arr_lists.append(
            context.get_constant_generic(builder, types.List(t), t_arr_list)
        )

    parent = context.get_constant_null(types.pyobject)
    t_len = context.get_constant(
        types.int64, 0 if len(pyval.arrays) == 0 else len(pyval.arrays[0])
    )
    return lir.Constant.literal_struct(arr_lists + [parent, t_len])


def get_init_table_output_type(table_type, to_str_if_dict_t):
    out_table_type = unwrap_typeref(table_type)
    assert isinstance(out_table_type, TableType), "table type or typeref expected"
    assert is_overload_constant_bool(to_str_if_dict_t), (
        "constant to_str_if_dict_t expected"
    )

    # convert dictionary-encoded string arrays to regular string arrays
    if is_overload_true(to_str_if_dict_t):
        out_table_type = to_str_arr_if_dict_array(out_table_type)

    return out_table_type


@intrinsic(prefer_literal=True)
def init_table(typingctx, table_type, to_str_if_dict_t):
    """initialize a table object with same structure as input table without setting it's
    array blocks (to be set later)

    NOTE: currently, to_str_if_dict_t is only set to True in decode_if_dict_table, and
    our gatherv implementation.
    If you're writing code that calls this function with to_str_if_dict_t==True,
    be sure to properly handle the case where the input table has a column of string,
    and dict_encoded string. These two blocks in the input table will be mapped to
    the same block in the output table.
    """
    out_table_type = get_init_table_output_type(table_type, to_str_if_dict_t)

    def codegen(context, builder, sig, args):
        table = cgutils.create_struct_proxy(out_table_type)(context, builder)
        for t, blk in out_table_type.type_to_blk.items():
            null_list = context.get_constant_null(types.List(t))
            setattr(table, f"block_{blk}", null_list)
        return table._getvalue()

    sig = out_table_type(table_type, to_str_if_dict_t)
    return sig, codegen


@intrinsic
def init_table_from_lists(typingctx, tuple_of_lists_type, table_type):
    """initialize a table object with table_type and list of arrays
    provided by the tuple of lists, tuple_of_lists_type.
    """
    assert isinstance(tuple_of_lists_type, types.BaseTuple), "Tuple of data expected"
    # Use a map to ensure the tuple and table ordering match
    tuple_map = {}
    for i, typ in enumerate(tuple_of_lists_type):
        assert isinstance(typ, types.List), "Each tuple element must be a list"
        tuple_map[typ.dtype] = i
    table_typ = (
        table_type.instance_type
        if isinstance(table_type, types.TypeRef)
        else table_type
    )
    assert isinstance(table_typ, TableType), "table type expected"

    def codegen(context, builder, sig, args):
        tuple_of_lists, _ = args
        table = cgutils.create_struct_proxy(table_typ)(context, builder)
        for t, blk in table_typ.type_to_blk.items():
            idx = tuple_map[t]
            # Get tuple_of_lists[idx]
            tuple_sig = signature(
                types.List(t), tuple_of_lists_type, types.literal(idx)
            )
            tuple_args = (tuple_of_lists, idx)
            list_elem = numba.cpython.tupleobj.static_getitem_tuple(
                context, builder, tuple_sig, tuple_args
            )
            setattr(table, f"block_{blk}", list_elem)
        return table._getvalue()

    sig = table_typ(tuple_of_lists_type, table_type)
    return sig, codegen


@intrinsic(prefer_literal=True)
def get_table_block(typingctx, table_type, blk_type):
    """get array list for a type block of table"""
    assert isinstance(table_type, TableType), "table type expected"
    assert_bodo_error(is_overload_constant_int(blk_type))
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
def ensure_table_unboxed(typingctx, table_type, used_cols_typ):
    """make all used in columns of table are unboxed.
    Throw an error if column array is null and there is no parent to unbox from

    used_cols is a set of columns that are used or None.
    """

    def codegen(context, builder, sig, args):
        # No need to unbox if already unboxed eagerly (improves compilation time)
        if bodo.hiframes.boxing.UNBOX_DATAFRAME_EAGERLY:
            return

        table_arg, used_col_set = args

        use_all = used_cols_typ == types.none
        if not use_all:
            set_inst = numba.cpython.setobj.SetInstance(
                context, builder, types.Set(types.int64), used_col_set
            )

        table = cgutils.create_struct_proxy(sig.args[0])(context, builder, table_arg)
        for t, blk in table_type.type_to_blk.items():
            n_arrs = context.get_constant(
                types.int64, len(table_type.block_to_arr_ind[blk])
            )
            # lower array of array indices for block to use within the loop
            # using array since list doesn't have constant lowering
            arr_inds = context.make_constant_array(
                builder,
                types.Array(types.int64, 1, "C"),
                # On windows np.array defaults to the np.int32 for integers.
                # As a result, we manually specify int64 during the array
                # creation to keep the lowered constant consistent with the
                # expected type.
                np.array(table_type.block_to_arr_ind[blk], dtype=np.int64),
            )
            arr_inds_struct = context.make_array(types.Array(types.int64, 1, "C"))(
                context, builder, arr_inds
            )
            arr_list = getattr(table, f"block_{blk}")
            with cgutils.for_range(builder, n_arrs) as loop:
                i = loop.index
                # get array index in "arrays"
                arr_ind = _getitem_array_single_int(
                    context,
                    builder,
                    types.int64,
                    types.Array(types.int64, 1, "C"),
                    arr_inds_struct,
                    i,
                )
                unbox_sig = types.none(
                    table_type, types.List(t), types.int64, types.int64
                )
                unbox_args = (table_arg, arr_list, i, arr_ind)
                if use_all:
                    # If we use all columns avoid generating control flow.
                    ensure_column_unboxed_codegen(
                        context, builder, unbox_sig, unbox_args
                    )
                else:
                    # If we need to check the column we generate an if.
                    use_col = set_inst.contains(arr_ind)
                    with builder.if_then(use_col):
                        ensure_column_unboxed_codegen(
                            context, builder, unbox_sig, unbox_args
                        )

    assert isinstance(table_type, TableType), "table type expected"
    sig = types.none(table_type, used_cols_typ)
    return sig, codegen


@intrinsic
def ensure_column_unboxed(typingctx, table_type, arr_list_t, ind_t, arr_ind_t):
    """make sure column of table is unboxed
    Throw an error if column array is null and there is no parent to unbox from
    table_type: table containing the parent structure to unbox
    arr_list_t: list of arrays that might be updated by the unboxing, list containing relevant column
    ind_t: index into arr_list_t where relevant column can be found (physical index of the column inside the list)
    arr_ind_t: index into the table where the relevant column is found (logical index of the column,
               i.e. column number in the original DataFrame)
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

    # No need to unbox if already unboxed eagerly (improves compilation time)
    if bodo.hiframes.boxing.UNBOX_DATAFRAME_EAGERLY:
        return

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


@intrinsic(prefer_literal=True)
def set_table_block(typingctx, table_type, arr_list_type, blk_type):
    """set table block and return a new table object"""
    assert isinstance(table_type, TableType), "table type expected"
    assert isinstance(arr_list_type, types.List), "list type expected"
    assert_bodo_error(is_overload_constant_int(blk_type), "blk should be const int")
    blk = get_overload_const_int(blk_type)

    def codegen(context, builder, sig, args):
        table_arg, arr_list_arg, _ = args
        in_table = cgutils.create_struct_proxy(table_type)(context, builder, table_arg)
        setattr(in_table, f"block_{blk}", arr_list_arg)
        return impl_ret_borrowed(context, builder, table_type, in_table._getvalue())

    sig = table_type(table_type, arr_list_type, blk_type)
    return sig, codegen


@intrinsic
def set_table_len(typingctx, table_type, l_type):
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
def set_table_parent(typingctx, out_table_type, in_table_type):
    """set out_table parent to the in_table's parent and return a new table object"""
    assert isinstance(in_table_type, TableType), "table type expected"
    assert isinstance(out_table_type, TableType), "table type expected"

    def codegen(context, builder, sig, args):
        out_table_arg, in_table_arg = args
        in_table = cgutils.create_struct_proxy(in_table_type)(
            context, builder, in_table_arg
        )
        out_table = cgutils.create_struct_proxy(out_table_type)(
            context, builder, out_table_arg
        )
        out_table.parent = in_table.parent
        context.nrt.incref(builder, types.pyobject, out_table.parent)
        return impl_ret_borrowed(
            context, builder, out_table_type, out_table._getvalue()
        )

    sig = out_table_type(out_table_type, in_table_type)
    return sig, codegen


@intrinsic(prefer_literal=True)
def alloc_list_like(typingctx, list_type, len_type, to_str_if_dict_t):
    """
    allocate a list with same type and size as input list but filled with null values
    """
    out_list_type = (
        list_type.instance_type if isinstance(list_type, types.TypeRef) else list_type
    )
    assert isinstance(out_list_type, types.List), "list type or typeref expected"
    assert isinstance(len_type, types.Integer), "integer type expected"
    assert is_overload_constant_bool(to_str_if_dict_t), (
        "constant to_str_if_dict_t expected"
    )

    # convert dictionary-encoded string arrays to regular string arrays
    if is_overload_true(to_str_if_dict_t):
        out_list_type = types.List(to_str_arr_if_dict_array(out_list_type.dtype))

    def codegen(context, builder, sig, args):
        size = args[1]
        _, out_arr_list = ListInstance.allocate_ex(
            context, builder, out_list_type, size
        )
        out_arr_list.size = size
        return out_arr_list.value

    sig = out_list_type(list_type, len_type, to_str_if_dict_t)
    return sig, codegen


@intrinsic
def alloc_empty_list_type(typingctx, size_typ, data_typ):
    """
    allocate a list with a given size and data type filled
    with null values.
    """
    assert isinstance(size_typ, types.Integer), "Size must be an integer"
    dtype = unwrap_typeref(data_typ)
    list_type = types.List(dtype)

    def codegen(context, builder, sig, args):
        size, _ = args
        _, out_arr_list = ListInstance.allocate_ex(context, builder, list_type, size)
        out_arr_list.size = size
        return out_arr_list.value

    sig = list_type(size_typ, data_typ)
    return sig, codegen


def _get_idx_num_true(idx):  # pragma: no cover
    pass


@overload(_get_idx_num_true)
def overload_get_idx_num_true(idx, n):
    """get length of boolean array or slice index"""
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        return lambda idx, n: idx.sum()  # pragma: no cover

    assert isinstance(idx, types.SliceType), "slice index expected"

    def impl(idx, n):  # pragma: no cover
        slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
        return numba.cpython.unicode._slice_span(slice_idx)

    return impl


def table_filter(T, idx, used_cols=None):
    pass


def gen_table_filter_impl(T, idx, used_cols=None):
    """Function that filters table using input boolean array or slice.
       If used_cols is passed, only used columns are written in output.

    Args:
        T (TableType): Table to be filtered
        idx (types.Array(types.bool_, 1, "C") | bodo.BooleanArrayType
            | types.SliceType): Index for filtering
        used_cols (types.TypeRef(bodo.MetaType)
            | types.none): If not None, the columns that should be selected
            to filter. This is filled by the table column deletion steps.

    Returns:
        TableType: The filtered table.
    """
    from bodo.utils.conversion import ensure_contig_if_np

    glbls = {
        "init_table": init_table,
        "get_table_block": get_table_block,
        "ensure_column_unboxed": ensure_column_unboxed,
        "set_table_block": set_table_block,
        "set_table_len": set_table_len,
        "alloc_list_like": alloc_list_like,
        "_get_idx_num_true": _get_idx_num_true,
        "ensure_contig_if_np": ensure_contig_if_np,
        "set_wrapper": set_wrapper,
    }
    if not is_overload_none(used_cols):
        # Get the MetaType from the TypeRef
        used_cols_type = used_cols.instance_type
        used_cols_data = np.array(used_cols_type.meta, dtype=np.int64)
        glbls["used_cols_vals"] = used_cols_data
        kept_blks = {T.block_nums[i] for i in used_cols_data}
    else:
        used_cols_data = None

    func_text = "def table_filter_func(T, idx, used_cols=None):\n"
    func_text += "  T2 = init_table(T, False)\n"
    func_text += "  l = 0\n"

    # set table length using index value and return if no table column is used
    if used_cols_data is not None and len(used_cols_data) == 0:
        # avoiding _get_idx_num_true in the general case below since it has extra overhead
        func_text += "  l = _get_idx_num_true(idx, len(T))\n"
        func_text += "  T2 = set_table_len(T2, l)\n"
        func_text += "  return T2\n"
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        return loc_vars["table_filter_func"]

    if used_cols_data is not None:
        func_text += "  used_set = set_wrapper(used_cols_vals)\n"
    for blk in T.type_to_blk.values():
        func_text += f"  arr_list_{blk} = get_table_block(T, {blk})\n"
        func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_{blk}, len(arr_list_{blk}), False)\n"
        if used_cols_data is None or blk in kept_blks:
            # Check if anything from this type is live. If not skip generating
            # the loop code.
            glbls[f"arr_inds_{blk}"] = np.array(T.block_to_arr_ind[blk], dtype=np.int64)
            func_text += f"  for i in range(len(arr_list_{blk})):\n"
            func_text += f"    arr_ind_{blk} = arr_inds_{blk}[i]\n"
            if used_cols_data is not None:
                func_text += f"    if arr_ind_{blk} not in used_set: continue\n"
            func_text += (
                f"    ensure_column_unboxed(T, arr_list_{blk}, i, arr_ind_{blk})\n"
                f"    out_arr_{blk} = ensure_contig_if_np(arr_list_{blk}[i][idx])\n"
                f"    l = len(out_arr_{blk})\n"
                f"    out_arr_list_{blk}[i] = out_arr_{blk}\n"
            )
        func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
    func_text += "  T2 = set_table_len(T2, l)\n"
    func_text += "  return T2\n"

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["table_filter_func"]


@infer_global(table_filter)
class TableFilterInfer(AbstractTemplate):
    """Typer for table_filter"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(table_filter)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        # output has same type as input
        out_type = folded_args[0]
        return signature(out_type, *folded_args).replace(pysig=pysig)


TableFilterInfer._no_unliteral = True


@lower_builtin(table_filter, types.VarArg(types.Any))
def lower_table_filter(context, builder, sig, args):
    """lower table_filter() using gen_table_filter_impl above"""
    impl = gen_table_filter_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def table_local_filter(T, idx, used_cols=None):
    """
    Function that filters table using input boolean array or slice
    only locally. This version is specially treated in Bodo passes.
    See table_filter for argument and return types.
    """


@infer_global(table_local_filter)
class TableLocalFilterInfer(AbstractTemplate):
    """Typer for table_local_filter"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(table_local_filter)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        # output has same type as input
        out_type = folded_args[0]
        return signature(out_type, *folded_args).replace(pysig=pysig)


TableLocalFilterInfer._no_unliteral = True


@lower_builtin(table_local_filter, types.VarArg(types.Any))
def lower_table_local_filter(context, builder, sig, args):
    """lower table_local_filter() using gen_table_filter_impl above"""
    impl = gen_table_filter_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def table_subset(T, idx, copy_arrs, used_cols=None):
    pass


def gen_table_subset_impl(T, idx, copy_arrs, used_cols=None):
    """Selects a subset of arrays/columns
    from a table using a list of integer column numbers.
    To minimize the size of the IR the integers are passed
    via a MetaType TypeRef.

    Args:
        T (TableType): Original table for taking a subset of arrays
        idx (TypeRef(MetaType)): Container for the constant list of columns
            that should be selected
        used_cols (TypeRef(MetaType) | None): Columns determined to be needed
            after table_column_del pass. This is used to further filter idx
            into only the columns that are actually needed.
    """
    # Extract the actual columns to select.
    cols_subset = list(idx.instance_type.meta)
    out_arr_typs = tuple(np.array(T.arr_types, dtype=object)[cols_subset])
    out_table_typ = TableType(out_arr_typs)
    if not is_overload_constant_bool(copy_arrs):
        raise_bodo_error("table_subset(): copy_arrs must be a constant")
    make_copy = is_overload_true(copy_arrs)
    glbls = {
        "init_table": init_table,
        "get_table_block": get_table_block,
        "ensure_column_unboxed": ensure_column_unboxed,
        "set_table_block": set_table_block,
        "set_table_len": set_table_len,
        "alloc_list_like": alloc_list_like,
        "out_table_typ": out_table_typ,
        "set_wrapper": set_wrapper,
    }
    # Determine which columns to prune if used_cols is included.
    # Note these are the column numbers of the output table, not
    # the input table.
    if not is_overload_none(used_cols):
        kept_cols = used_cols.instance_type.meta
        kept_cols_set = set(kept_cols)
        glbls["kept_cols"] = np.array(kept_cols, np.int64)
        skip_cols = True
    else:
        skip_cols = False
    # Compute a mapping of columns before removing
    # dropped columns.
    moved_cols_map = dict(enumerate(cols_subset))

    func_text = "def table_subset(T, idx, copy_arrs, used_cols=None):\n"
    func_text += "  T2 = init_table(out_table_typ, False)\n"
    # Length of the table cannot change.
    func_text += "  T2 = set_table_len(T2, len(T))\n"

    # set table length using index value and return if no table column is used
    if skip_cols and len(kept_cols_set) == 0:
        # If all columns are dead just return the table.
        func_text += "  return T2\n"
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        return loc_vars["table_subset"]

    if skip_cols:
        # Create a set for filtering
        func_text += "  kept_cols_set = set_wrapper(kept_cols)\n"

    # Use the output table to only generate code per output
    # type. Here we iterate over the output type and list of
    # arrays and copy the arrays from the original table. Here
    # is some pseudo-code (omitting many details like unboxing).

    #   for typ in out_types:
    #       out_arrs = alloc_list(typ)
    #       in_arrs = get_block(table, typ)
    #       for idx in range(len(out_arrs)):
    #           # Map the physical offset to df colnum
    #           logical_idx = get_logical_idx(idx)
    #           # We skip any columns that are dead
    #           if logical_idx not dead:
    #               # Map the new actual index back to the old
    #               orig_idx = get_orig_idx(idx)
    #               out_arrs[idx] = in_arrs[orig_idx]
    #       set_block(out_table, out_arrs, typ)
    #   return out_table
    #
    # Note: We must iterate over the output because the input
    # could specify duplicate columns.
    #
    # If an entire output type is pruned we generate the list and omit
    # the for loop.

    for typ, blk in out_table_typ.type_to_blk.items():
        # Since we have a subset of columns this must exist.
        orig_table_blk = T.type_to_blk[typ]
        func_text += f"  arr_list_{blk} = get_table_block(T, {orig_table_blk})\n"
        func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_{blk}, {len(out_table_typ.block_to_arr_ind[blk])}, False)\n"
        has_match = True
        if skip_cols:
            # If we are deleting columns we may be able to skip the loop.
            block_colnums_set = set(out_table_typ.block_to_arr_ind[blk])
            matched_cols = block_colnums_set & kept_cols_set
            # Determine if any column matches. If not we don't need
            # the loop.
            has_match = len(matched_cols) > 0
        if has_match:
            glbls[f"out_arr_inds_{blk}"] = np.array(
                out_table_typ.block_to_arr_ind[blk], dtype=np.int64
            )
            # We iterate over the output array because there are always <=
            # the number of elements in the input array list.
            func_text += f"  for i in range(len(out_arr_list_{blk})):\n"
            func_text += f"    out_arr_ind_{blk} = out_arr_inds_{blk}[i]\n"
            if skip_cols:
                func_text += (
                    f"    if out_arr_ind_{blk} not in kept_cols_set: continue\n"
                )

            # Determine a mapping back to the old index from the new index in case
            # columns are reordered. Since we may need to unbox the column, we need
            # both a logical index (DataFrame column number) and a physical index
            # (offset in the list of arrays).
            in_logical_idx = []
            in_physical_idx = []
            for num in out_table_typ.block_to_arr_ind[blk]:
                in_logical = moved_cols_map[num]
                in_logical_idx.append(in_logical)
                in_physical = T.block_offsets[in_logical]
                in_physical_idx.append(in_physical)
            glbls[f"in_logical_idx_{blk}"] = np.array(in_logical_idx, dtype=np.int64)
            glbls[f"in_physical_idx_{blk}"] = np.array(in_physical_idx, dtype=np.int64)
            func_text += f"    logical_idx_{blk} = in_logical_idx_{blk}[i]\n"
            func_text += f"    physical_idx_{blk} = in_physical_idx_{blk}[i]\n"
            # We must check if we need to unbox the original column because DataFrames
            # do lazy unboxing.
            func_text += f"    ensure_column_unboxed(T, arr_list_{blk}, physical_idx_{blk}, logical_idx_{blk})\n"
            suffix = ".copy()" if make_copy else ""
            func_text += f"    out_arr_list_{blk}[i] = arr_list_{blk}[physical_idx_{blk}]{suffix}\n"
        func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
    func_text += "  return T2\n"

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["table_subset"]


@infer_global(table_subset)
class TableSubsetInfer(AbstractTemplate):
    """Typer for table_subset"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(table_subset)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        T = folded_args[0]
        idx = folded_args[1]
        cols_subset = list(unwrap_typeref(idx).meta)
        out_arr_typs = tuple(np.array(T.arr_types, dtype=object)[cols_subset])
        out_table_type = TableType(out_arr_typs)
        return signature(out_table_type, *folded_args).replace(pysig=pysig)


TableSubsetInfer._no_unliteral = True


@lower_builtin(table_subset, types.VarArg(types.Any))
def lower_table_subset(context, builder, sig, args):
    """lower table_subset() using gen_table_subset_impl above"""
    impl = gen_table_subset_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def table_filter_equiv(self, scope, equiv_set, loc, args, kws):
    """output of table_filter has the same number of columns as input table.
    If there is an empty slice its also the same length."""
    var = args[0]
    if equiv_set.has_shape(var):
        if guard(is_whole_slice, self.typemap, self.func_ir, args[1]):
            return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
        return ArrayAnalysis.AnalyzeResult(
            shape=(None, equiv_set.get_shape(var)[1]), pre=[]
        )
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_table_table_filter = table_filter_equiv
ArrayAnalysis._analyze_op_call_bodo_hiframes_table_table_local_filter = (
    table_filter_equiv
)


def table_subset_equiv(self, scope, equiv_set, loc, args, kws):
    """output of table_subset has the same length as input table"""
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(
            shape=(equiv_set.get_shape(var)[0], None), pre=[]
        )
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_table_table_subset = table_subset_equiv


def gen_str_and_dict_enc_cols_to_one_block_fn_txt(
    in_table_type, out_table_type, glbls, is_gatherv=False
):
    """
    Helper function to avoid duplication in decode_if_dict_table and our gatherv impl.

    Given the input and output table types. Generates a func text that extracts
    the string/encoded string columns from input table "T" and uses those values to set a singular
    string block in output table "T2". If is_gatherv is set, performs a gather on the input arrays
    before setting the output table.

    Here is an example generated functext:

    input_str_arr_list = get_table_block(T, 0)
    input_dict_enc_str_arr_list = get_table_block(T, 1)
    out_arr_list_0 = alloc_list_like(input_str_arr_list, 4, True)
    for input_str_ary_idx, output_str_arr_offset in enumerate(output_table_str_arr_offsets_in_combined_block):
        arr_ind_str = arr_inds_0[input_str_ary_idx]
        ensure_column_unboxed(T, input_str_arr_list, input_str_ary_idx, arr_ind_str)
        out_arr_str = input_str_arr_list[input_str_ary_idx]
        out_arr_str = bodo.gatherv(out_arr_str, allgather, warn_if_rep, root)
        out_arr_list_0[output_str_arr_offset] = out_arr_str
    for input_dict_enc_str_ary_idx, output_dict_enc_str_arr_offset in enumerate(output_table_dict_enc_str_arr_offsets_in_combined_block):
        arr_ind_dict_enc_str = arr_inds_1[input_dict_enc_str_ary_idx]
        ensure_column_unboxed(T, input_dict_enc_str_arr_list, input_dict_enc_str_ary_idx, arr_ind_dict_enc_str)
        out_arr_dict_enc_str = decode_if_dict_array(input_dict_enc_str_arr_list[input_dict_enc_str_ary_idx])
        out_arr_dict_enc_str = bodo.gatherv(out_arr_dict_enc_str, allgather, warn_if_rep, root)
        out_arr_list_0[output_dict_enc_str_arr_offset] = out_arr_dict_enc_str
    T2 = set_table_block(T2, out_arr_list_0, 0)

    Args:
        in_table_type (TableType): The input table, from which to source the string columns.
                                   Must contain a string and encoded string block
        out_table_type (TableType): The out table, whose columns will be set.
                                    Must contain a string block
        glbls (dict): The dict of global variable to be used when executing this functext.
                      keys 'arr_inds_(input table string block)', 'arr_inds_(input table dict encoded string block)',
                      'decode_if_dict_array', and 'output_table_str_arr_idxs_in_combined_block' are added by this function
        is_gatherv (optional bool): If set, the functext will perform a gatherv on the input arrays
                                    before setting the output table.
    """

    # NOTE: there are several asserts throughout this file which we expect to never run.
    # These asserts are needed, as any incorrect assumptions in this portion of the code
    # can lead to very difficult to debug errors.

    assert (
        bodo.string_array_type in in_table_type.type_to_blk
        and bodo.string_array_type in in_table_type.type_to_blk
    ), (
        f"Error in gen_str_and_dict_enc_cols_to_one_block_fn_txt: Table type {in_table_type} does not contain both a string, and encoded string column"
    )

    input_string_ary_blk = in_table_type.type_to_blk[bodo.string_array_type]
    input_dict_encoded_string_ary_blk = in_table_type.type_to_blk[
        bodo.dict_str_arr_type
    ]

    # NOTE: Care must be taken to ensure that arrays are placed into the output block
    # in the correct ordering. The ordering in which the arrays will be placed into the physical table
    # is dependent on the logical ordering of the columns in the tabletype's arr_types attribute.
    # For example, if the input table has array_types in the order
    # (String, dict_enc, String, dict_enc)
    # which correspond to the arrays:
    # (str_col_0, dict_col_0, str_col_1, dict_col_1)
    # The locations of each of the arrays in input table would be (In the format (name, block, index_into_block))
    # (str_col_0, block_str, 0), (dict_col_0, block_dict_str, 0), (str_col_1, block_str, 1), (dict_col_1, block_dict_str, 1)

    # After the cast, the array types of the output table are
    # (String, String, String, String)
    # Therefore, we expect the offsets in the output table to be (In the format (name, block, index_into_block))
    # (str_col_0, block_str, 0), (dict_col_0, block_str, 1), (str_col_1, block_str, 2), (dict_col_1, block_str, 3)

    # In order to handle this, we need to get the expected ordering from
    # the type itself.

    logical_string_array_idxs = in_table_type.block_to_arr_ind.get(input_string_ary_blk)
    logical_dic_enc_string_array_idxs = in_table_type.block_to_arr_ind.get(
        input_dict_encoded_string_ary_blk
    )

    string_array_physical_offset_in_output_block = []
    dict_enc_string_array_physical_offset_in_output_block = []

    cur_str_ary_idx = 0
    cur_dict_enc_str_ary_idx = 0

    # Iterate over the logical indices of the string/dictionary columns
    # for each physical index in the output string block, we assign it to whichever
    # column appears first in the logical ordering
    for cur_output_table_offset in range(
        len(logical_string_array_idxs) + len(logical_dic_enc_string_array_idxs)
    ):
        if cur_str_ary_idx == len(logical_string_array_idxs):
            # We've already assigned all the string columns
            dict_enc_string_array_physical_offset_in_output_block.append(
                cur_output_table_offset
            )
            continue
        elif cur_dict_enc_str_ary_idx == len(logical_dic_enc_string_array_idxs):
            # We've already assigned all the dict columns
            string_array_physical_offset_in_output_block.append(cur_output_table_offset)
            continue

        cur_string_array_logical_idx = logical_string_array_idxs[cur_str_ary_idx]
        cur_dict_enc_string_array_logical_idx = logical_dic_enc_string_array_idxs[
            cur_dict_enc_str_ary_idx
        ]

        if cur_string_array_logical_idx < cur_dict_enc_string_array_logical_idx:
            # If the lowest logical column number for string columns is lower than the lowest
            # dict encoded string array, assign the string column the physical offset
            string_array_physical_offset_in_output_block.append(cur_output_table_offset)
            cur_str_ary_idx += 1
        else:
            # Otherwise, assign it to the dict encoded column
            dict_enc_string_array_physical_offset_in_output_block.append(
                cur_output_table_offset
            )
            cur_dict_enc_str_ary_idx += 1

    assert "output_table_str_arr_offsets_in_combined_block" not in glbls, (
        "Error in gen_str_and_dict_enc_cols_to_one_block_fn_txt: key 'output_table_str_arr_idxs_in_combined_block' already present as a global variable"
    )
    glbls["output_table_str_arr_offsets_in_combined_block"] = np.array(
        string_array_physical_offset_in_output_block
    )
    assert "output_table_dict_enc_str_arr_offsets_in_combined_block" not in glbls, (
        "Error in gen_str_and_dict_enc_cols_to_one_block_fn_txt: key 'output_table_str_arr_idxs_in_combined_block' already present as a global variable"
    )
    glbls["output_table_dict_enc_str_arr_offsets_in_combined_block"] = np.array(
        dict_enc_string_array_physical_offset_in_output_block
    )

    glbls["decode_if_dict_array"] = decode_if_dict_array

    out_table_block = out_table_type.type_to_blk[bodo.string_array_type]

    assert f"arr_inds_{input_string_ary_blk}" not in glbls, (
        f"Error in gen_str_and_dict_enc_cols_to_one_block_fn_txt: arr_inds_{input_string_ary_blk} already present in global variables"
    )
    glbls[f"arr_inds_{input_string_ary_blk}"] = np.array(
        in_table_type.block_to_arr_ind[input_string_ary_blk], dtype=np.int64
    )

    assert f"arr_inds_{input_dict_encoded_string_ary_blk}" not in glbls, (
        f"Error in gen_str_and_dict_enc_cols_to_one_block_fn_txt: arr_inds_{input_dict_encoded_string_ary_blk} already present in global variables"
    )
    glbls[f"arr_inds_{input_dict_encoded_string_ary_blk}"] = np.array(
        in_table_type.block_to_arr_ind[input_dict_encoded_string_ary_blk],
        dtype=np.int64,
    )

    func_text = f"  input_str_arr_list = get_table_block(T, {input_string_ary_blk})\n"
    func_text += f"  input_dict_enc_str_arr_list = get_table_block(T, {input_dict_encoded_string_ary_blk})\n"

    func_text += f"  out_arr_list_{out_table_block} = alloc_list_like(input_str_arr_list, {len(string_array_physical_offset_in_output_block) + len(dict_enc_string_array_physical_offset_in_output_block)}, True)\n"

    # handle string arrays
    func_text += "  for input_str_ary_idx, output_str_arr_offset in enumerate(output_table_str_arr_offsets_in_combined_block):\n"
    func_text += (
        f"    arr_ind_str = arr_inds_{input_string_ary_blk}[input_str_ary_idx]\n"
    )
    func_text += "    ensure_column_unboxed(T, input_str_arr_list, input_str_ary_idx, arr_ind_str)\n"
    func_text += "    out_arr_str = input_str_arr_list[input_str_ary_idx]\n"
    if is_gatherv:
        func_text += "    out_arr_str = bodo.gatherv(out_arr_str, allgather, warn_if_rep, root)\n"

    func_text += (
        f"    out_arr_list_{out_table_block}[output_str_arr_offset] = out_arr_str\n"
    )

    # handle dict encoded string arrays
    func_text += "  for input_dict_enc_str_ary_idx, output_dict_enc_str_arr_offset in enumerate(output_table_dict_enc_str_arr_offsets_in_combined_block):\n"
    func_text += f"    arr_ind_dict_enc_str = arr_inds_{input_dict_encoded_string_ary_blk}[input_dict_enc_str_ary_idx]\n"
    func_text += "    ensure_column_unboxed(T, input_dict_enc_str_arr_list, input_dict_enc_str_ary_idx, arr_ind_dict_enc_str)\n"
    func_text += "    out_arr_dict_enc_str = decode_if_dict_array(input_dict_enc_str_arr_list[input_dict_enc_str_ary_idx])\n"
    if is_gatherv:
        func_text += "    out_arr_dict_enc_str = bodo.gatherv(out_arr_dict_enc_str, allgather, warn_if_rep, root)\n"

    func_text += f"    out_arr_list_{out_table_block}[output_dict_enc_str_arr_offset] = out_arr_dict_enc_str\n"

    func_text += f"  T2 = set_table_block(T2, out_arr_list_{out_table_block}, {out_table_block})\n"

    return func_text


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def decode_if_dict_table(T):
    """
    Takes a Table that may contain a dict_str_array_type
    and converts the dict_str_array to a regular string
    array. This is used as a fallback when some operations
    aren't supported on dictionary arrays.

    Note: This current DOES NOT currently handle dead columns
    and is currently only used in situations where we are certain
    all columns are alive (i.e. before write).
    """
    func_text = "def impl(T):\n"
    func_text += "  T2 = init_table(T, True)\n"
    func_text += "  l = len(T)\n"
    glbls = {
        "init_table": init_table,
        "get_table_block": get_table_block,
        "ensure_column_unboxed": ensure_column_unboxed,
        "set_table_block": set_table_block,
        "set_table_len": set_table_len,
        "alloc_list_like": alloc_list_like,
        "decode_if_dict_array": decode_if_dict_array,
    }

    out_table_type = bodo.hiframes.table.get_init_table_output_type(T, True)
    input_table_has_str_and_dict_encoded_str = (
        bodo.string_array_type in T.type_to_blk
        and bodo.dict_str_arr_type in T.type_to_blk
    )

    # In the case that we have both normal and dict encoded string arrays, we need to do
    # special handling, as the dict encoded string arrays will be converted to string arrays,
    # so the two blocks for string arrays and dict encoded string arrays in the input table
    # will be stored in only one block in the output table
    if input_table_has_str_and_dict_encoded_str:
        func_text += gen_str_and_dict_enc_cols_to_one_block_fn_txt(
            T, out_table_type, glbls
        )

    for typ, input_blk in T.type_to_blk.items():
        # Skip these blocks if we handle them above
        if input_table_has_str_and_dict_encoded_str and typ in (
            bodo.string_array_type,
            bodo.dict_str_arr_type,
        ):
            continue

        # Output block num may be different from input block num in certain cases.
        # Specifically, if the input table has a string and dict encoded string block,
        # which will be fused into one block in the output table.
        if typ == bodo.dict_str_arr_type:
            assert bodo.string_array_type in out_table_type.type_to_blk, (
                "Error in decode_if_dict_table: If encoded string type is present in the input, then non-encoded string type should be present in the output"
            )
            output_blk = out_table_type.type_to_blk[bodo.string_array_type]
        else:
            assert typ in out_table_type.type_to_blk, (
                "Error in decode_if_dict_table: All non-encoded string types present in the input should be present in the output"
            )
            output_blk = out_table_type.type_to_blk[typ]

        glbls[f"arr_inds_{input_blk}"] = np.array(
            T.block_to_arr_ind[input_blk], dtype=np.int64
        )
        func_text += f"  arr_list_{input_blk} = get_table_block(T, {input_blk})\n"
        func_text += f"  out_arr_list_{input_blk} = alloc_list_like(arr_list_{input_blk}, len(arr_list_{input_blk}), True)\n"
        func_text += f"  for i in range(len(arr_list_{input_blk})):\n"
        func_text += f"    arr_ind_{input_blk} = arr_inds_{input_blk}[i]\n"
        func_text += f"    ensure_column_unboxed(T, arr_list_{input_blk}, i, arr_ind_{input_blk})\n"
        func_text += (
            f"    out_arr_{input_blk} = decode_if_dict_array(arr_list_{input_blk}[i])\n"
        )
        func_text += f"    out_arr_list_{input_blk}[i] = out_arr_{input_blk}\n"
        func_text += (
            f"  T2 = set_table_block(T2, out_arr_list_{input_blk}, {output_blk})\n"
        )
    func_text += "  T2 = set_table_len(T2, l)\n"
    func_text += "  return T2\n"

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl"]


@overload(operator.getitem, no_unliteral=True, inline="always")
def overload_table_getitem(T, idx):
    if not isinstance(T, TableType):
        return
    return lambda T, idx: table_filter(T, idx)  # pragma: no cover


@intrinsic
def init_runtime_table_from_lists(typingctx, arr_list_tup_typ, nrows_typ):
    """
    Takes a list of arrays and the length of each array and creates a Python
    Table from this list (without making a copy).

    The number of arrays in this table is NOT known at compile time.
    """
    assert isinstance(arr_list_tup_typ, types.BaseTuple), (
        "init_runtime_table_from_lists requires a tuple of list of arrays"
    )
    if isinstance(arr_list_tup_typ, types.UniTuple):
        # When running type inference, we may require further transformations
        # to determine the types of the lists. As a result we return None
        # when the type is undefined to trigger another pass.
        if arr_list_tup_typ.dtype.dtype == types.undefined:
            return
        arr_list_typs = [arr_list_tup_typ.dtype.dtype] * len(arr_list_tup_typ)
    else:
        arr_list_typs = []
        for typ in arr_list_tup_typ:
            if typ.dtype == types.undefined:
                return
            arr_list_typs.append(typ.dtype)
    assert isinstance(nrows_typ, types.Integer), (
        "init_runtime_table_from_lists requires an integer length"
    )

    def codegen(context, builder, sig, args):
        (arr_list_tup, nrows) = args
        table = cgutils.create_struct_proxy(table_type)(context, builder)
        # Update the table length. This is assume to be 0 if the arr_list is empty
        table.len = nrows
        # Store each array list in the table and increment its refcount.
        arr_lists = cgutils.unpack_tuple(builder, arr_list_tup)
        for i, arr_list in enumerate(arr_lists):
            setattr(table, f"block_{i}", arr_list)
            context.nrt.incref(builder, types.List(arr_list_typs[i]), arr_list)
        return table._getvalue()

    table_type = TableType(tuple(arr_list_typs), True)
    sig = table_type(arr_list_tup_typ, nrows_typ)
    return sig, codegen


def _to_arr_if_series(t):
    """returns underlying array data type if 't' is a SeriesType, otherwise returns 't'"""
    return t.data if isinstance(t, SeriesType) else t


def create_empty_table(table_type):
    pass


def gen_create_empty_table_impl(table_type):
    """Implement of create_empty_table(), which create a table with the
    required table type and all columns with length 0.

    Note: This doesn't support dictionary encoded arrays. If you call this
    function make sure you convert any dictionary array types to regular
    string arrays.

    Args:
        table_type (TableType): The type of the output table.
    Returns:
        TableType: An allocated table with length 0.
    """
    out_table_type = unwrap_typeref(table_type)
    glbls = {
        "alloc_list_like": alloc_list_like,
        "alloc_type": alloc_type,
        "init_table": init_table,
        "set_table_len": set_table_len,
        "set_table_block": set_table_block,
    }
    func_text = "def impl_create_empty_table(table_type):\n"
    func_text += "  table = init_table(table_type, False)\n"
    func_text += "  table = set_table_len(table, 0)\n"
    for typ, blk in out_table_type.type_to_blk.items():
        glbls[f"arr_typ_{blk}"] = typ
        glbls[f"arr_list_typ_{blk}"] = types.List(typ)
        n_arrs = len(out_table_type.block_to_arr_ind[blk])
        func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_typ_{blk}, {n_arrs}, False)\n"
        # assign arrays that come from input table.
        func_text += f"  for i in range(len(out_arr_list_{blk})):\n"
        func_text += (
            f"    out_arr_list_{blk}[i] = alloc_type(0, arr_typ_{blk}, (-1,))\n"
        )
        func_text += f"  table = set_table_block(table, out_arr_list_{blk}, {blk})\n"

    func_text += "  return table\n"
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl_create_empty_table"]


@infer_global(create_empty_table)
class CreateEmptyTableInfer(AbstractTemplate):
    """Typer for create_empty_table"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(create_empty_table)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        out_table_type = unwrap_typeref(folded_args[0])
        return signature(out_table_type, *folded_args).replace(pysig=pysig)


@lower_builtin(create_empty_table, types.Any)
def lower_create_empty_table(context, builder, sig, args):
    """lower create_empty_table() using gen_create_empty_table_impl() above"""
    impl = gen_create_empty_table_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def create_empty_table_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for create_empty_table(). Output table has
    length 0.
    """
    return ArrayAnalysis.AnalyzeResult(shape=(ir.Const(0, loc), None), pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_table_create_empty_table = (
    create_empty_table_equiv
)


def logical_table_to_table(
    in_table_t,
    extra_arrs_t,
    in_col_inds_t,
    n_table_cols_t,
    out_table_type_t=None,
    used_cols=None,
):
    pass


def gen_logical_table_to_table_impl(
    in_table_t,
    extra_arrs_t,
    in_col_inds_t,
    n_table_cols_t,
    out_table_type_t=None,
    used_cols=None,
):
    """Convert a logical table formed by table and array data into an actual table.
    For example, assuming T1 has 3 columns:
        in_table_t = T1
        extra_arrs_t = (A, B)
        in_col_inds_t = (2, 4, 3, 1)
        output = (T1_2, B, A, T1_1)

    Args:
        in_table_t (TableType|types.Tuple(array)): input table data (could be a
            TableType or a tuple of arrays)
        extra_arrs_t (types.Tuple(array)): extra arrays in addition to previous table
            argument to form the full logical input table
        in_col_inds_t (MetaType(list[int])): logical indices in input for each
            output table column. Actual input data is a table that has logical
            columns 0..n_in_table_arrs-1, and regular arrays that have the rest of data
            (n_in_table_arrs, n_in_table_arrs+1, ...).
        n_table_cols_t (int): number of logical input columns in input table in_table_t.
            Necessary since in_table_t may be set to none if dead. The number becomes
            a constant in dataframe pass but not during initial typing.
        out_table_type_t (TypeRef): output table type. Necessary since some input data
            may be set to None if dead. Provided in table column del pass after
            optimization but not before.
        used_cols (TypeRef(MetaType) | None): Output columns that are actually used
            as found by table_column_del pass. If None, all columns are used.

    Returns:
        TableType: converted table
    """
    in_col_inds = in_col_inds_t.instance_type.meta
    assert isinstance(in_table_t, (TableType, types.BaseTuple, types.NoneType)), (
        "logical_table_to_table: input table must be a TableType or tuple of arrays or None (for dead table)"
    )

    glbls = {"set_wrapper": set_wrapper}
    # prune columns if used_cols is provided
    if not is_overload_none(used_cols):
        kept_cols = set(used_cols.instance_type.meta)
        glbls["kept_cols"] = np.array(list(kept_cols), np.int64)
        skip_cols = True
    else:
        kept_cols = set(np.arange(len(in_col_inds)))
        skip_cols = False

    # BodoSQL may pass Series data by mistake so need conversion to arrays
    extra_arrs_no_series = ", ".join(
        f"get_series_data(extra_arrs_t[{i}])"
        if isinstance(extra_arrs_t[i], SeriesType)
        else f"extra_arrs_t[{i}]"
        for i in range(len(extra_arrs_t))
    )
    extra_arrs_no_series = (
        f"({extra_arrs_no_series}{',' if len(extra_arrs_t) == 1 else ''})"
    )

    # handle array-only input data
    if isinstance(in_table_t, (types.BaseTuple, types.NoneType)):
        return _logical_tuple_table_to_table_codegen(
            in_table_t,
            extra_arrs_t,
            in_col_inds,
            kept_cols,
            n_table_cols_t,
            out_table_type_t,
            extra_arrs_no_series,
        )

    # at this point in_table is provided as a TableType but extra arrays may be None
    n_in_table_arrs = len(in_table_t.arr_types)

    out_table_type = (
        TableType(
            tuple(
                in_table_t.arr_types[i]
                if i < n_in_table_arrs
                else _to_arr_if_series(extra_arrs_t.types[i - n_in_table_arrs])
                for i in in_col_inds
            )
        )
        if is_overload_none(out_table_type_t)
        else unwrap_typeref(out_table_type_t)
    )

    # Add the globals needed for the 0 column path.
    glbls.update(
        {
            "init_table": init_table,
            "set_table_len": set_table_len,
            "out_table_type": out_table_type,
            "set_wrapper": set_wrapper,
        }
    )

    func_text = "def impl_logical_table_to_table(in_table_t, extra_arrs_t, in_col_inds_t, n_table_cols_t, out_table_type_t=None, used_cols=None):\n"
    if any(isinstance(t, SeriesType) for t in extra_arrs_t.types):
        func_text += f"  extra_arrs_t = {extra_arrs_no_series}\n"
    func_text += "  T1 = in_table_t\n"
    func_text += "  T2 = init_table(out_table_type, False)\n"
    func_text += "  T2 = set_table_len(T2, len(T1))\n"

    # If all columns are dead just return the table (only table length is used).
    if skip_cols and len(kept_cols) == 0:
        func_text += "  return T2\n"
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        return loc_vars["impl_logical_table_to_table"]

    # Create a set for column filtering
    if skip_cols:
        func_text += "  kept_cols_set = set_wrapper(kept_cols)\n"

    for typ, blk in out_table_type.type_to_blk.items():
        glbls[f"arr_list_typ_{blk}"] = types.List(typ)
        n_arrs = len(out_table_type.block_to_arr_ind[blk])
        func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_typ_{blk}, {n_arrs}, False)\n"
        # assign arrays that come from input table.
        if typ in in_table_t.type_to_blk:
            in_table_blk = in_table_t.type_to_blk[typ]
            # for each array in output block, get block offset and array index in input
            # table (if it's coming from the input table)
            in_blk_idxs = []
            in_arr_inds = []
            for out_arr_ind in out_table_type.block_to_arr_ind[blk]:
                in_arr_ind = in_col_inds[out_arr_ind]
                if in_arr_ind < n_in_table_arrs:
                    in_blk_idxs.append(in_table_t.block_offsets[in_arr_ind])
                    in_arr_inds.append(in_arr_ind)
                else:
                    in_blk_idxs.append(-1)
                    in_arr_inds.append(-1)
            glbls[f"in_idxs_{blk}"] = np.array(in_blk_idxs, np.int64)
            glbls[f"in_arr_inds_{blk}"] = np.array(in_arr_inds, np.int64)
            if skip_cols:
                glbls[f"out_arr_inds_{blk}"] = np.array(
                    out_table_type.block_to_arr_ind[blk], dtype=np.int64
                )
            func_text += f"  in_arr_list_{blk} = get_table_block(T1, {in_table_blk})\n"
            func_text += f"  for i in range(len(out_arr_list_{blk})):\n"
            func_text += f"    in_offset_{blk} = in_idxs_{blk}[i]\n"
            func_text += f"    if in_offset_{blk} == -1:\n"
            func_text += "      continue\n"
            func_text += f"    in_arr_ind_{blk} = in_arr_inds_{blk}[i]\n"
            if skip_cols:
                func_text += (
                    f"    if out_arr_inds_{blk}[i] not in kept_cols_set: continue\n"
                )
            func_text += f"    ensure_column_unboxed(T1, in_arr_list_{blk}, in_offset_{blk}, in_arr_ind_{blk})\n"
            func_text += (
                f"    out_arr_list_{blk}[i] = in_arr_list_{blk}[in_offset_{blk}]\n"
            )
        # assign arrays that come from extra arrays
        for i, out_arr_ind in enumerate(out_table_type.block_to_arr_ind[blk]):
            if out_arr_ind not in kept_cols:
                continue
            in_arr_ind = in_col_inds[out_arr_ind]
            if in_arr_ind >= n_in_table_arrs:
                func_text += f"  out_arr_list_{blk}[{i}] = extra_arrs_t[{in_arr_ind - n_in_table_arrs}]\n"

        func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"

    func_text += "  return T2\n"

    glbls.update(
        {
            "alloc_list_like": alloc_list_like,
            "set_table_block": set_table_block,
            "get_table_block": get_table_block,
            "ensure_column_unboxed": ensure_column_unboxed,
            "get_series_data": bodo.hiframes.pd_series_ext.get_series_data,
        }
    )

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl_logical_table_to_table"]


def _logical_tuple_table_to_table_codegen(
    in_table_t,
    extra_arrs_t,
    in_col_inds,
    kept_cols,
    n_table_cols_t,
    out_table_type_t,
    extra_arrs_no_series,
):
    """generate a function for logical table to table conversion when input "table" is
    a tuple of arrays. See logical_table_to_table() for details.

    Args:
        in_table_t (types.Tuple(array)): input table data (tuple of arrays)
        extra_arrs_t (types.Tuple(array)): extra arrays in addition to previous table
            argument to form the full logical input table
        in_col_inds_t (list[int]): logical indices in input for each output table column
        kept_cols (set[int]): set of output table column indices that are actually used
        n_table_cols_t (int): number of logical input columns in input table in_table_t.
            Necessary since in_table_t may be set to none if dead. The number becomes
            a constant in dataframe pass but not during initial typing.
        out_table_type_t (TypeRef): output table type. Necessary since some input data
            may be set to None if dead. Provided in table column del pass after
            optimization but not before.

    Returns:
        function: generated function for logical table to table conversion
    """
    n_in_table_arrs = (
        get_overload_const_int(n_table_cols_t)
        if is_overload_constant_int(n_table_cols_t)
        else len(in_table_t.types)
    )

    out_table_type = (
        TableType(
            tuple(
                in_table_t.types[i]
                if i < n_in_table_arrs
                else _to_arr_if_series(extra_arrs_t.types[i - n_in_table_arrs])
                for i in in_col_inds
            )
        )
        if is_overload_none(out_table_type_t)
        else unwrap_typeref(out_table_type_t)
    )

    # find an array in input data to use for setting length of output that is not None
    # after dead column elimination. There is always at least one array that is not None
    # since logical_table_to_table() would be eliminated otherwise.
    len_arr = None
    if not is_overload_none(in_table_t):
        for i, t in enumerate(in_table_t.types):
            if t != types.none:
                len_arr = f"in_table_t[{i}]"
                break

    if len_arr is None:
        for i, t in enumerate(extra_arrs_t.types):
            if t != types.none:
                len_arr = f"extra_arrs_t[{i}]"
                break

    assert len_arr is not None, "no array found in input data"

    func_text = "def impl(in_table_t, extra_arrs_t, in_col_inds_t, n_table_cols_t, out_table_type_t=None, used_cols=None):\n"
    if any(isinstance(t, SeriesType) for t in extra_arrs_t.types):
        func_text += f"  extra_arrs_t = {extra_arrs_no_series}\n"
    func_text += "  T1 = in_table_t\n"
    func_text += "  T2 = init_table(out_table_type, False)\n"
    func_text += f"  T2 = set_table_len(T2, len({len_arr}))\n"
    glbls = {}

    for typ, blk in out_table_type.type_to_blk.items():
        glbls[f"arr_list_typ_{blk}"] = types.List(typ)
        n_arrs = len(out_table_type.block_to_arr_ind[blk])
        func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_typ_{blk}, {n_arrs}, False)\n"

        # assign arrays that come from tuple of arrays or extra arrays
        for i, out_arr_ind in enumerate(out_table_type.block_to_arr_ind[blk]):
            if out_arr_ind not in kept_cols:
                continue
            in_arr_ind = in_col_inds[out_arr_ind]
            if in_arr_ind < n_in_table_arrs:
                func_text += f"  out_arr_list_{blk}[{i}] = T1[{in_arr_ind}]\n"
            else:
                func_text += f"  out_arr_list_{blk}[{i}] = extra_arrs_t[{in_arr_ind - n_in_table_arrs}]\n"

        func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"

    func_text += "  return T2\n"

    glbls.update(
        {
            "init_table": init_table,
            "alloc_list_like": alloc_list_like,
            "set_table_block": set_table_block,
            "set_table_len": set_table_len,
            "out_table_type": out_table_type,
            "get_series_data": bodo.hiframes.pd_series_ext.get_series_data,
        }
    )

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl"]


@infer_global(logical_table_to_table)
class LogicalTableToTableInfer(AbstractTemplate):
    """Typer for logical_table_to_table"""

    def generic(self, args, kws):
        kws = dict(kws)
        out_table_type_t = get_call_expr_arg(
            "logical_table_to_table",
            args,
            kws,
            4,
            "out_table_type_t",
            default=types.none,
        )
        # out_table_type is provided in table column del pass after optimization
        if is_overload_none(out_table_type_t):
            in_table_t = get_call_expr_arg(
                "logical_table_to_table", args, kws, 0, "in_table_t"
            )
            extra_arrs_t = get_call_expr_arg(
                "logical_table_to_table", args, kws, 1, "extra_arrs_t"
            )
            in_col_inds_t = get_call_expr_arg(
                "logical_table_to_table", args, kws, 2, "in_col_inds_t"
            )
            n_table_cols_t = get_call_expr_arg(
                "logical_table_to_table", args, kws, 3, "n_table_cols_t"
            )
            # Make sure inputs are arrays (also catches types.unknown)
            for t in extra_arrs_t:
                assert is_array_typ(t, True), (
                    f"logical_table_to_table: array type expected but got {t}"
                )

            in_col_inds = unwrap_typeref(in_col_inds_t).meta
            # handle array-only input data
            if isinstance(in_table_t, (types.BaseTuple, types.NoneType)):
                n_in_table_arrs = (
                    get_overload_const_int(n_table_cols_t)
                    if is_overload_constant_int(n_table_cols_t)
                    else len(in_table_t.types)
                )
                out_table_type = TableType(
                    tuple(
                        in_table_t.types[i]
                        if i < n_in_table_arrs
                        else _to_arr_if_series(extra_arrs_t.types[i - n_in_table_arrs])
                        for i in in_col_inds
                    )
                )
            else:
                n_in_table_arrs = len(in_table_t.arr_types)
                out_table_type = TableType(
                    tuple(
                        in_table_t.arr_types[i]
                        if i < n_in_table_arrs
                        else _to_arr_if_series(extra_arrs_t.types[i - n_in_table_arrs])
                        for i in in_col_inds
                    )
                )
        else:
            out_table_type = unwrap_typeref(out_table_type_t)

        pysig = numba.core.utils.pysignature(logical_table_to_table)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(out_table_type, *folded_args).replace(pysig=pysig)


LogicalTableToTableInfer._no_unliteral = True


@lower_builtin(logical_table_to_table, types.VarArg(types.Any))
def lower_logical_table_to_table(context, builder, sig, args):
    """lower logical_table_to_table() using gen_logical_table_to_table_impl above"""
    impl = gen_logical_table_to_table_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def logical_table_to_table_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for logical_table_to_table().
    Output table has the same length as input table or arrays.
    """
    table_var = args[0]
    extra_arrs_var = args[1]

    if equiv_set.has_shape(table_var):
        # Table can be an empty tuple, which sometimes has an empty tuple and is
        # not correct.
        equivs = equiv_set.get_shape(table_var)
        if equivs:
            return ArrayAnalysis.AnalyzeResult(shape=(equivs[0], None), pre=[])

    if equiv_set.has_shape(extra_arrs_var):
        # Extra arrays can be an empty tuple, which may have an empty
        # tuple and is not correct.
        equivs = equiv_set.get_shape(extra_arrs_var)
        if equivs:
            return ArrayAnalysis.AnalyzeResult(shape=(equivs[0], None), pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_table_logical_table_to_table = (
    logical_table_to_table_equiv
)


def alias_ext_logical_table_to_table(lhs_name, args, alias_map, arg_aliases):
    """add aliasing info for logical_table_to_table(). Output table reuses input table
    and arrays.
    """
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    numba.core.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("logical_table_to_table", "bodo.hiframes.table")
] = alias_ext_logical_table_to_table


def generate_empty_table_with_rows(n_rows):  # pragma: no cover
    pass


@overload(generate_empty_table_with_rows)
def overload_generate_empty_table_with_rows(n_rows):
    """
    Generates a table with no columns and n_rows rows.
    """
    table_typ = TableType(())

    def impl(n_rows):  # pragma: no cover
        T = init_table(table_typ, False)
        result = set_table_len(T, n_rows)
        return result

    return impl
