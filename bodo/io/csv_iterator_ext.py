"""
Class information for DataFrame iterators returned by pd.read_csv. This is used
to handle situations in which pd.read_csv is used to return chunks with separate
read calls instead of just a single read.
"""

import llvmlite.binding as ll
import numba
import numpy as np  # noqa
import pandas as pd  # noqa
from llvmlite import ir as lir
from numba.core import cgutils, ir_utils, types
from numba.core.imputils import RefType, impl_ret_borrowed, iternext_impl
from numba.core.typing.templates import signature
from numba.extending import intrinsic, lower_builtin, models, register_model

import bodo
import bodo.ir.connector
import bodo.ir.csv_ext
from bodo import objmode  # noqa
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.table import Table, TableType  # noqa
from bodo.io import csv_json_reader
from bodo.ir.csv_ext import _gen_read_csv_objmode, astype  # noqa
from bodo.utils.typing import ColNamesMetaType
from bodo.utils.utils import (
    check_java_installation,  # noqa
    sanitize_varname,
)

ll.add_symbol(
    "update_csv_reader", csv_json_reader.get_function_address("update_csv_reader")
)
ll.add_symbol(
    "initialize_csv_reader",
    csv_json_reader.get_function_address("initialize_csv_reader"),
)


class CSVIteratorType(types.SimpleIteratorType):
    """
    Iterator class used with pd.read_csv should return an iterator
    instead of a DataFrame.
    """

    def __init__(
        self,
        df_type,
        out_colnames,
        out_types,
        usecols,
        sep,
        index_ind,
        index_arr_typ,
        index_name,
        escapechar,
        storage_options,
    ):
        assert isinstance(df_type, DataFrameType), "CSVIterator must return a DataFrame"
        name = f"CSVIteratorType({df_type}, {out_colnames}, {out_types}, {usecols}, {sep}, {index_ind}, {index_arr_typ}, {index_name}, {escapechar})"
        super(types.SimpleIteratorType, self).__init__(name)
        self._yield_type = df_type
        # CSV info used to call pd.read_csv
        self._out_colnames = out_colnames
        self._out_types = out_types
        self._usecols = usecols
        self._sep = sep
        # Which column should be the index. If None we generate a
        # RangeIndex
        self._index_ind = index_ind
        self._index_arr_typ = index_arr_typ
        self._index_name = index_name
        self._escapechar = escapechar
        self._storage_options = storage_options

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(CSVIteratorType)
class CSVIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("csv_reader", types.stream_reader_type),
            # "index" is the number of rows read so far,
            # which is needed to create RangeIndex of output
            # DataFrame if necessary.
            # "index" is not used for iterator state
            # - csv_reader keeps iterator state.
            ("index", types.EphemeralPointer(types.uintp)),
        ]
        super().__init__(dmm, fe_type, members)


@lower_builtin("getiter", CSVIteratorType)
def getiter_csv_iterator(context, builder, sig, args):
    """
    Iternext call for CSVIterator. This generates the code to
    initialize the csv_reader and the index.
    """
    # Initialize the CSV_reader. This is used to indicate that
    # update_csv_reader is being called the first time.
    iterator_struct = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
        ],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="initialize_csv_reader"
    )
    csv_reader_struct = cgutils.create_struct_proxy(types.stream_reader_type)(
        context, builder, value=iterator_struct.csv_reader
    )
    builder.call(fn_tp, [csv_reader_struct.pyobj])
    # Initialize the index. TODO: Does this change with nrows?
    builder.store(context.get_constant(types.uint64, 0), iterator_struct.index)

    # simply return the iterator
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


@lower_builtin("iternext", CSVIteratorType)
@iternext_impl(RefType.NEW)
def iternext_csv_iterator(context, builder, sig, args, result):
    """
    Iternext call for CSVIterator. This generates the code to
    return the dataframe, sets the output into result, and
    updates the index.
    """
    [iterty] = sig.args
    [iter_arg] = args

    # Update the C++ state
    iterator_struct = cgutils.create_struct_proxy(iterty)(
        context, builder, value=iter_arg
    )
    fnty = lir.FunctionType(
        lir.IntType(1),
        [
            lir.IntType(8).as_pointer(),
        ],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="update_csv_reader"
    )
    csv_reader_struct = cgutils.create_struct_proxy(types.stream_reader_type)(
        context, builder, value=iterator_struct.csv_reader
    )
    is_valid = builder.call(fn_tp, [csv_reader_struct.pyobj])
    # Set the valid bit based on if C++ has more to read
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        index = builder.load(iterator_struct.index)
        # Perform the actual csv_read
        tuple_typ = types.Tuple([sig.return_type.first_type, types.int64])
        impl = gen_read_csv_objmode(sig.args[0])
        read_csv_sig = signature(tuple_typ, types.stream_reader_type, types.int64)
        ret_tuple = context.compile_internal(
            builder, impl, read_csv_sig, [iterator_struct.csv_reader, index]
        )
        out_df, chunk_len = cgutils.unpack_tuple(builder, ret_tuple)
        # Taken from Numba.
        # We pass the "nsw" flag in the hope that LLVM understands the index
        # never changes sign.  Unfortunately this doesn't always work
        new_start = builder.add(index, chunk_len, flags=["nsw"])
        builder.store(new_start, iterator_struct.index)

        result.yield_(out_df)


@intrinsic
def init_csv_iterator(typingctx, csv_reader, csv_iterator_typeref):
    """Create a CSV iterator with the provided csv_reader. This reader contains
    the relevant info to call pd.read_csv. csv_iterator_typeref is a typeref
    used to set the output type.
    """

    def codegen(context, builder, signature, args):
        iterator = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        context.nrt.incref(builder, signature.args[0], args[0])
        iterator.csv_reader = args[0]
        zero = context.get_constant(types.uintp, 0)
        iterator.index = cgutils.alloca_once_value(builder, zero)
        return iterator._getvalue()

    assert isinstance(csv_iterator_typeref, types.TypeRef), (
        "Initializing a csv iterator requires a typeref"
    )
    ret_typ = csv_iterator_typeref.instance_type
    sig = signature(ret_typ, csv_reader, csv_iterator_typeref)
    return sig, codegen


def gen_read_csv_objmode(csv_iterator_type):
    """
    Function that generates the objmode call for pd.read_csv.
    This creates a two functions, 1 that contains the objmode and
    is executed with numba.njit and a wrapper function that constructs
    the return DataFrame (which is not a dispatcher).

    This function assumes that the f_reader has already been updated by
    the iterator and that this function will only be called if there
    is more data to return.
    """
    func_text = "def read_csv_objmode(f_reader):\n"
    santized_cnames = [sanitize_varname(c) for c in csv_iterator_type._out_colnames]
    call_id = ir_utils.next_label()
    glbls = globals()
    out_types = csv_iterator_type._out_types
    glbls[f"table_type_{call_id}"] = TableType(tuple(out_types))
    glbls["idx_array_typ"] = csv_iterator_type._index_arr_typ
    # We don't yet support removing columns from the source in the chunksize case
    out_used_cols = list(range(len(csv_iterator_type._usecols)))
    func_text += _gen_read_csv_objmode(
        csv_iterator_type._out_colnames,
        santized_cnames,
        out_types,
        csv_iterator_type._usecols,
        out_used_cols,
        csv_iterator_type._sep,
        csv_iterator_type._escapechar,
        csv_iterator_type._storage_options,
        call_id,
        glbls,
        parallel=False,
        check_parallel_runtime=True,
        idx_col_index=csv_iterator_type._index_ind,
        idx_col_typ=csv_iterator_type._index_arr_typ,
    )
    parallel_varname = bodo.ir.csv_ext._gen_parallel_flag_name(santized_cnames)
    total_names = (
        ["T"]
        + (["idx_arr"] if csv_iterator_type._index_ind is not None else [])
        + [parallel_varname]
    )
    func_text += f"  return {', '.join(total_names)}"
    glbls = globals()
    # TODO: Provide specific globals after Numba's #3355 is resolved
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    read_csv_objmode = loc_vars["read_csv_objmode"]
    objmode_func = numba.njit(read_csv_objmode)
    # Is this necessary?
    bodo.ir.csv_ext.compiled_funcs.append(objmode_func)
    # Seems like you can't call withObjMode from compile_internal, so
    # we use two different functions.
    wrapper_func_text = "def read_func(reader, local_start):\n"
    wrapper_func_text += f"  {', '.join(total_names)} = objmode_func(reader)\n"
    index_ind = csv_iterator_type._index_ind
    if index_ind is None:
        # Manually create the parallel range index at runtime
        # since this won't run through distributed pass
        wrapper_func_text += "  local_len = len(T)\n"
        wrapper_func_text += "  total_size = local_len\n"
        wrapper_func_text += f"  if ({parallel_varname}):\n"
        wrapper_func_text += "    local_start = local_start + bodo.libs.distributed_api.dist_exscan(local_len, _op)\n"
        wrapper_func_text += (
            "    total_size = bodo.libs.distributed_api.dist_reduce(local_len, _op)\n"
        )
        index_arg = "bodo.hiframes.pd_index_ext.init_range_index(local_start, local_start + local_len, 1, None)"
    else:
        # Total is garbage if we have an index column but must be returned
        wrapper_func_text += "  total_size = 0\n"
        index_arg = f"bodo.utils.conversion.convert_to_index({total_names[1]}, {csv_iterator_type._index_name!r})"
    wrapper_func_text += f"  return (bodo.hiframes.pd_dataframe_ext.init_dataframe(({total_names[0]},), {index_arg}, __col_name_meta_value_read_csv_objmode), total_size)\n"
    exec(
        wrapper_func_text,
        {
            "bodo": bodo,
            "objmode_func": objmode_func,
            "_op": np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value),
            "__col_name_meta_value_read_csv_objmode": ColNamesMetaType(
                csv_iterator_type.yield_type.columns
            ),
        },
        loc_vars,
    )
    return loc_vars["read_func"]
