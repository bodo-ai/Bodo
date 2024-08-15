# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements wrappers to call the C++ BodoSQL array kernels for listagg.
"""

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic

import bodo
from bodo.hiframes.pd_dataframe_ext import get_dataframe_all_data
from bodo.hiframes.table import logical_table_to_table
from bodo.libs import listagg
from bodo.libs.array import arr_info_list_to_table, array_to_info, delete_table
from bodo.libs.distributed_api import allgatherv
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.utils.typing import (
    get_overload_const_bool,
    get_overload_const_str,
    get_overload_const_tuple,
    is_overload_constant_str,
    is_overload_constant_tuple,
    raise_bodo_error,
)

ll.add_symbol(
    "listagg_seq_py",
    listagg.listagg_seq_py,
)


@numba.generated_jit(nopython=True, no_unliteral=True)
def bodosql_listagg(
    df,
    agg_col,
    order_cols,
    ascending,
    na_position,
    separator="",
    is_parallel=False,
):
    """
    Common code to both sequential and parallel implementation of listagg.

    Basically performs checks, re-arranges the dataframe in the expected order,
    and calls the sequential/parallel impl.
    """
    is_parallel = get_overload_const_bool(is_parallel, "bodosql_listagg", "is_parallel")

    if not is_overload_constant_str(agg_col):
        raise_bodo_error(
            "Error in bodosql_listagg_seq: Aggregation column must be a string literal.",
        )
    else:
        agg_col = get_overload_const_str(agg_col)

    for required_const_tuple_arg, name in [
        (order_cols, "order_cols"),
        (ascending, "ascending"),
        (na_position, "na_position"),
    ]:
        if not is_overload_constant_tuple(required_const_tuple_arg):
            raise_bodo_error(
                f"Error in bodosql_listagg_seq: {name} must be constant tuple",
            )

    order_cols = get_overload_const_tuple(order_cols)
    ascending = get_overload_const_tuple(ascending)
    na_position = get_overload_const_tuple(na_position)

    assert len(ascending) == len(
        na_position
    ), "Internal error in bodosql_listagg_common: ascending and na_position must have same length"
    assert len(ascending) == len(
        order_cols
    ), "Internal error in bodosql_listagg_common: ascending and order_cols must have same length"

    for order_col in order_cols:
        if order_col not in df.columns:
            raise_bodo_error(
                "Error in bodosql_listagg_seq: Order columns must be contained in dataframe",
            )

    needed_cols = tuple(list(order_cols) + [agg_col])

    na_position_real = tuple([na_pos == "last" for na_pos in na_position])

    # only used in parallel version
    # DataFrame should be ordered orderCols, agg_col before calling dropna
    dropna_subset = (df.columns[-1],)

    def bodosql_listagg_impl(
        df,
        agg_col,
        order_cols,
        ascending,
        na_position,
        separator="",
        is_parallel=False,
    ):  # pragma: no cover
        if is_parallel:
            # There are two reasons why we would want to drop NA values from the input table:
            # 1. Certain operations in the C++ code that will be faster
            #    --- can potentially avoid some null checking while iterating over the rows of the table
            #    --- size of dataframe is smaller which means faster sorting/iteration, etc.
            # 2. The allgather shuffle will be faster if we don't have to send the un-needed NA values
            #    --- only relevant in parallel mode
            #
            # While I'm not certain if the costs of dropping NA in the sequential case outweigh the benefits,
            # I'm pretty sure that the costs of dropping NA in the parallel case are worth it.
            # TODO: actually verify this (https://bodo.atlassian.net/browse/BSE-720)

            reduced_df = df.dropna(subset=dropna_subset)
            reordered_df = reduced_df.loc[:, needed_cols]
            # NOTE FOR POTENTIAL FUTURE OPTIMIZATION:
            # We could do a local sort on each input table prior to the allGather,
            # and use the fact that all the individual input tables are sorted
            # to perform the final sort on the gathered table
            # much more quickly. For now, it seems easier just to do the full sort on
            # each rank.
            gathered_df = allgatherv(reordered_df)
            return bodosql_listagg_seq(
                gathered_df, ascending, na_position_real, separator
            )
        else:
            return bodosql_listagg_seq(
                df.loc[:, needed_cols],
                ascending,
                na_position_real,
                separator,
            )

    return bodosql_listagg_impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def bodosql_listagg_seq(df, ascending, na_position, separator=""):
    """
    Sequential implementation of listagg.
    Assumes that the input dataframe is replicated, and the columns are ordered: (order_col_1, order_col_2, ... agg_col)
    """

    ascending = get_overload_const_tuple(ascending)
    na_position = get_overload_const_tuple(na_position)

    func_text = "def impl(df, ascending, na_position, separator=''):\n"

    # NOTE: adding extra False to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "  ascending = np.array([{}], dtype=np.bool_)\n".format(
        ", ".join([str(i) for i in ascending] + ["False"])
    )
    # NOTE: adding extra False to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "  na_position = np.array([{}], dtype=np.bool_)\n".format(
        ", ".join([str(i) for i in na_position] + ["False"])
    )

    in_col_inds = bodo.utils.typing.MetaType(tuple(range(len(df.columns))))
    num_table_cols = len(df.columns)
    func_text += f"  py_table = logical_table_to_table(get_dataframe_all_data(df), (), in_col_inds, {num_table_cols})\n"
    func_text += "  cpp_table = py_table_to_cpp_table(py_table, py_table_typ)\n"
    func_text += f"  out_str = listagg_seq_cpp(cpp_table, unicode_to_utf8(separator), {len(ascending)}, ascending.ctypes, na_position.ctypes, len(separator))\n"
    func_text += "  return out_str\n"

    tmp_locs = {}
    tmp_glbls = {
        "bodo": bodo,
        "listagg_seq_cpp": listagg_seq_cpp,
        "py_table_to_cpp_table": bodo.hiframes.pd_dataframe_ext.py_table_to_cpp_table,
        "py_table_typ": df.table_type
        if df.is_table_format
        else bodo.TableType(df.data),
        "unicode_to_utf8": unicode_to_utf8,
        "np": np,
        "arr_info_list_to_table": arr_info_list_to_table,
        "array_to_info": array_to_info,
        "in_col_inds": in_col_inds,
        "logical_table_to_table": logical_table_to_table,
        "get_dataframe_all_data": get_dataframe_all_data,
        "delete_table": delete_table,
    }

    exec(func_text, tmp_glbls, tmp_locs)
    impl = tmp_locs["impl"]
    return impl


@intrinsic
def listagg_seq_cpp(
    typingctx,
    table_t,
    separator_t,
    num_order_cols_t,
    ascending_t,
    na_position_t,
    separator_len_t,
):
    """
    Interface to listagg_seq_py function in C++ library. We're using an intrinsic
    instead of a direct external call in order to handle allocating
    an int pointer for the output string size.
    """

    def codegen(context, builder, sig, args):  # pragma: no cover
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # char* return
            [
                lir.IntType(8).as_pointer(),  # table_info *raw_in_table
                lir.IntType(8).as_pointer(),  # char *separator
                lir.IntType(32),  # int num_order_cols
                lir.IntType(8).as_pointer(),  # bool *ascending
                lir.IntType(8).as_pointer(),  # bool *na_position
                # TODO: can I make this unsigned?
                lir.IntType(64),  # int64_t separator_len
                lir.IntType(
                    64
                ).as_pointer(),  # int64_t *output_string_size TODO: how do I make this unsigned?
            ],
        )
        listagg_fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="listagg_seq_py"
        )

        output_string_size = cgutils.alloca_once(builder, lir.IntType(64))

        data_ptr = builder.call(listagg_fn_tp, args + (output_string_size,))

        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

        # decode_utf8 version
        decode_sig = bodo.string_type(types.voidptr, types.int64)
        ret = context.compile_internal(
            builder,
            lambda data, length: bodo.libs.str_arr_ext.decode_utf8(data, length),
            decode_sig,
            [data_ptr, builder.load(output_string_size)],
        )

        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

        return ret

    args_types = (
        types.voidptr,  # table_info *raw_in_table
        types.voidptr,  # char *separator
        types.int32,  # int num_order_cols
        types.voidptr,  # bool *ascending
        types.voidptr,  # bool *na_position
        types.int64,  # int64_t separator_len
        types.voidptr,  # int64_t *output_string_size
    )
    ret_type = string_type

    # TODO: what is recvr here? Do I need to set this to an actual non-None value?
    sig = numba.core.typing.templates.Signature(ret_type, args_types, None)

    return (
        sig,
        codegen,
    )
