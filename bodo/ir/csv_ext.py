import typing as pt
from collections import defaultdict

import numba
import numpy as np  # noqa
import pandas as pd  # noqa
from llvmlite import ir as lir
from mpi4py import MPI
from numba.core import cgutils, ir, ir_utils, typeinfer, types
from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes
from numba.extending import intrinsic

import bodo
import bodo.ir.connector
import bodo.user_logging
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.hiframes.table import Table, TableType  # noqa
from bodo.io.helpers import (
    get_storage_options_pyobject,
    storage_options_dict_type,
)
from bodo.ir.connector import Connector
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import StringArrayType, string_array_type
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.table_column_del_pass import (
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import (
    check_java_installation,  # noqa
    create_arg_hash,
    sanitize_varname,
)

if pt.TYPE_CHECKING:  # pragma: no cover
    from bodo.io.csv_iterator_ext import CSVIteratorType


class CsvReader(Connector):
    connector_typ = "csv"

    def __init__(
        self,
        file_name,
        df_out_varname: str,
        sep,
        out_table_col_names: list[str],
        out_vars: list[ir.Var],
        out_table_col_types: list[types.ArrayCompatible],
        usecols,
        loc,
        header,
        compression,
        nrows,
        skiprows,
        chunksize,
        chunk_iterator: pt.Optional["CSVIteratorType"],
        is_skiprows_list,
        low_memory,
        escapechar,
        storage_options=None,
        index_column_index=None,
        index_column_typ=types.none,
    ):
        self.file_name = file_name
        self.df_out_varname = df_out_varname  # used only for printing
        self.sep = sep
        self.out_table_col_names = out_table_col_names
        self.out_vars = out_vars
        self.out_table_col_types = out_table_col_types
        self.usecols = usecols
        self.loc = loc
        self.skiprows = skiprows
        self.nrows = nrows
        self.header = header
        self.compression = compression
        # If this value is not None, we return an iterator instead of a DataFrame.
        # When this happens the out_vars are a list with a single CSVReaderType.
        self.chunksize = chunksize
        self.chunk_iterator = chunk_iterator
        # skiprows list
        self.is_skiprows_list = is_skiprows_list
        self.pd_low_memory = low_memory
        self.escapechar = escapechar
        self.storage_options = storage_options
        self.index_column_index = index_column_index
        self.index_column_typ = index_column_typ
        # Columns within the output table type that are actually used.
        # These will be updated during optimzations and do not contain
        # the actual columns numbers that should be loaded. For more
        # information see 'csv_remove_dead_column'.
        self.out_used_cols = list(range(len(usecols)))

    def __repr__(self):  # pragma: no cover
        return f"{self.df_out_varname} = ReadCsv(file={self.file_name}, col_names={self.out_table_col_names}, col_types={self.out_table_col_types}, vars={self.out_vars}, nrows={self.nrows}, skiprows={self.skiprows}, chunksize={self.chunksize}, is_skiprows_list={self.is_skiprows_list}, pd_low_memory={self.pd_low_memory}, escapechar={self.escapechar}, storage_options={self.storage_options}, index_column_index={self.index_column_index}, index_colum_typ = {self.index_column_typ}, out_used_colss={self.out_used_cols})"

    def out_vars_and_types(self) -> list[tuple[str, types.Type]]:
        return (
            [(self.out_vars[0].name, self.chunk_iterator)]
            if self.is_streaming
            else [
                (self.out_vars[0].name, TableType(tuple(self.out_table_col_types))),
                (self.out_vars[1].name, self.index_column_typ),
            ]
        )


def check_node_typing(node, typemap):
    """
    Provides basic type checking for each relevant csv field. These only check values
    that can be passed as variables and constants are assumed to be checked in
    untyped_pass.
    """
    # Filename must be a string
    file_name_typ = typemap[node.file_name.name]
    if types.unliteral(file_name_typ) != types.unicode_type:
        raise BodoError(
            f"pd.read_csv(): 'filepath_or_buffer' must be a string. Found type: {file_name_typ}.",
            node.file_name.loc,
        )
    # Skip rows must be an integer, list of integers, or tuple of integers
    # If the value is a constant, we have already checked types in untyped pass.
    if not isinstance(node.skiprows, ir.Const):
        skiprows_typ = typemap[node.skiprows.name]
        if isinstance(skiprows_typ, types.Dispatcher):
            raise BodoError(
                "pd.read_csv(): 'skiprows' callable not supported yet.",
                node.file_name.loc,
            )
            # is_overload_constant_list
        elif (
            not isinstance(skiprows_typ, types.Integer)
            and not (
                isinstance(skiprows_typ, (types.List, types.Tuple))
                and isinstance(skiprows_typ.dtype, types.Integer)
            )
            and not (
                isinstance(
                    skiprows_typ, (types.LiteralList, bodo.utils.typing.ListLiteral)
                )
            )
        ):
            raise BodoError(
                f"pd.read_csv(): 'skiprows' must be an integer or list of integers. Found type {skiprows_typ}.",
                loc=node.skiprows.loc,
            )
        # Set flag for lists that are variables.
        elif isinstance(skiprows_typ, (types.List, types.Tuple)):
            node.is_skiprows_list = True
    # nrows must be an integer
    # If the value is an IR constant, then it is the default value so we don't need to check.
    if not isinstance(node.nrows, ir.Const):
        nrows_typ = typemap[node.nrows.name]
        if not isinstance(nrows_typ, types.Integer):
            raise BodoError(
                f"pd.read_csv(): 'nrows' must be an integer. Found type {nrows_typ}.",
                loc=node.nrows.loc,
            )


import llvmlite.binding as ll

from bodo.io import csv_json_reader

ll.add_symbol(
    "csv_file_chunk_reader",
    csv_json_reader.get_function_address("csv_file_chunk_reader"),
)


@intrinsic
def csv_file_chunk_reader(
    typingctx,
    fname_t,
    is_parallel_t,
    skiprows_t,
    nrows_t,
    header_t,
    compression_t,
    bucket_region_t,
    storage_options_t,
    chunksize_t,
    is_skiprows_list_t,
    skiprows_list_len_t,
    pd_low_memory_t,
):
    """
    Interface to csv_file_chunk_reader function in C++ library for creating
    the csv file reader.
    """
    # TODO: Update storage options to pyobject once the type is updated to do refcounting
    # properly.
    assert storage_options_t == storage_options_dict_type, (
        "Storage options don't match expected type"
    )

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # filename
                lir.IntType(1),  # is_parallel
                lir.IntType(8).as_pointer(),  # skiprows (array of int64_t)
                lir.IntType(64),  # nrows
                lir.IntType(1),  # header
                lir.IntType(8).as_pointer(),  # compressoin
                lir.IntType(8).as_pointer(),  # bucket_region
                lir.IntType(8).as_pointer(),  # storage_options dictionary
                lir.IntType(64),  # chunksize
                lir.IntType(1),  # is_skiprows_list
                lir.IntType(64),  # skiprows_list_len
                lir.IntType(1),  # pd_low_memory
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="csv_file_chunk_reader"
        )
        obj = builder.call(fn_tp, args)

        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        # csv_file_chunk_reader returns a pyobject. We need to wrap the result in the
        # proper return type and create a meminfo.
        ret = cgutils.create_struct_proxy(types.stream_reader_type)(context, builder)
        pyapi = context.get_python_api(builder)
        # borrows and manages a reference for obj (see comments in py_objs.py)
        ret.meminfo = pyapi.nrt_meminfo_new_from_pyobject(
            context.get_constant_null(types.voidptr), obj
        )
        ret.pyobj = obj
        # `nrt_meminfo_new_from_pyobject` increfs the object (holds a reference)
        # so need to decref since the object is not live anywhere else.
        pyapi.decref(obj)
        return ret._getvalue()

    return (
        types.stream_reader_type(
            types.voidptr,
            types.bool_,
            types.voidptr,  # skiprows (array of int64_t)
            types.int64,
            types.bool_,
            types.voidptr,
            types.voidptr,
            storage_options_dict_type,  # storage_options dictionary
            types.int64,  # chunksize
            types.bool_,  # is_skiprows_list
            types.int64,  # skiprows_list_len
            types.bool_,  # pd_low_memory
        ),
        codegen,
    )


def remove_dead_csv(
    csv_node: CsvReader,
    lives_no_aliases,
    lives,
    arg_aliases,
    alias_map,
    func_ir,
    typemap,
):
    """
    Function to determine to remove the returned variables
    once they are dead. This only removes whole variables, not sub-components
    like table columns.
    """
    if csv_node.chunksize is not None:
        # Chunksize only has 1 var
        iterator_var = csv_node.out_vars[0]
        if iterator_var.name not in lives:
            return None
    else:
        # Otherwise we have two variables.
        table_var = csv_node.out_vars[0]
        idx_var = csv_node.out_vars[1]

        # If both variables are dead, remove the node
        if table_var.name not in lives and idx_var.name not in lives:
            return None
        # If only the index variable is dead
        # update the fields in the node relating to the index column,
        # so that it doesn't get loaded from CSV
        elif idx_var.name not in lives:
            csv_node.index_column_index = None
            csv_node.index_column_typ = types.none
        # If the index variable is dead
        # update the fields in the node relating to the index column,
        # so that it doesn't get loaded from CSV
        elif table_var.name not in lives:
            csv_node.usecols = []
            csv_node.out_table_col_types = []
            csv_node.out_used_cols = []

    return csv_node


def csv_distributed_run(
    csv_node: CsvReader, array_dists, typemap, calltypes, typingctx, targetctx
):
    """
    Generate that actual code for this ReadCSV Node during distributed pass.
    This produces different code depending on if the read_csv call contains
    chunksize or not.
    """
    # skiprows as `ir.Const` indicates default value.
    # If it's a list, it will never be `ir.Const`
    skiprows_typ = (
        types.int64
        if isinstance(csv_node.skiprows, ir.Const)
        else types.unliteral(typemap[csv_node.skiprows.name])
    )
    if csv_node.chunksize is not None:
        # parallel read flag
        # Add debug info about column pruning. Chunksize doesn't yet prune
        # any columns.
        if bodo.user_logging.get_verbose_level() >= 1:
            msg = "Finish column pruning on read_csv node:\n%s\nColumns loaded %s\n"
            csv_source = csv_node.loc.strformat()
            csv_cols = csv_node.out_table_col_names
            bodo.user_logging.log_message("Column Pruning", msg, csv_source, csv_cols)

            # Log if any columns use dictionary encoded arrays.
            col_types = csv_node.out_table_col_types
            dict_encoded_cols = [
                c
                for i, c in enumerate(csv_node.out_table_col_names)
                if isinstance(col_types[i], bodo.libs.dict_arr_ext.DictionaryArrayType)
            ]
            if dict_encoded_cols:
                encoding_msg = "Finished optimized encoding on read_csv node:\n%s\nColumns %s using dictionary encoding to reduce memory usage.\n"
                bodo.user_logging.log_message(
                    "Dictionary Encoding",
                    encoding_msg,
                    csv_source,
                    dict_encoded_cols,
                )

        parallel = bodo.ir.connector.is_chunked_connector_table_parallel(
            csv_node, array_dists, "CSVReader"
        )

        # Iterator Case

        # Create a wrapper function that will be compiled. This will return
        # an iterator.
        func_text = "def csv_iterator_impl(fname, nrows, skiprows):\n"
        func_text += "    reader = _csv_reader_init(fname, nrows, skiprows)\n"
        func_text += "    iterator = init_csv_iterator(reader, csv_iterator_type)\n"
        loc_vars = {}
        from bodo.io.csv_iterator_ext import init_csv_iterator

        exec(func_text, {}, loc_vars)
        csv_iterator_impl = loc_vars["csv_iterator_impl"]

        # Generate an inner function to minimize the IR size.
        init_func_text = "def csv_reader_init(fname, nrows, skiprows):\n"

        # Appends func text to initialize a file stream reader.
        init_func_text += _gen_csv_file_reader_init(
            parallel,
            csv_node.header,
            csv_node.compression,
            csv_node.chunksize,
            csv_node.is_skiprows_list,
            csv_node.pd_low_memory,
            csv_node.storage_options,
        )
        init_func_text += "  return f_reader\n"
        exec(init_func_text, globals(), loc_vars)
        csv_reader_init = loc_vars["csv_reader_init"]

        # njit the function so it can be called by our outer function.
        # We keep track of the function for possible dynamic addresses
        jit_func = numba.njit(csv_reader_init)
        compiled_funcs.append(jit_func)

        # Compile the outer function into IR
        f_block = compile_to_numba_ir(
            csv_iterator_impl,
            {
                "_csv_reader_init": jit_func,
                "init_csv_iterator": init_csv_iterator,
                "csv_iterator_type": typemap[csv_node.out_vars[0].name],
            },
            typingctx=typingctx,
            targetctx=targetctx,
            # file_name, nrows, skiprows
            arg_typs=(string_type, types.int64, skiprows_typ),
            typemap=typemap,
            calltypes=calltypes,
        ).blocks.popitem()[1]

        # Replace the arguments with the values from the csv node
        replace_arg_nodes(
            f_block, [csv_node.file_name, csv_node.nrows, csv_node.skiprows]
        )
        # Replace the generated return statements with a node that returns
        # the csv iterator var.
        nodes = f_block.body[:-3]
        nodes[-1].target = csv_node.out_vars[0]

        return nodes

    # Default Case
    # Parallel is based on table + index var
    parallel = bodo.ir.connector.is_connector_table_parallel(
        csv_node, array_dists, typemap, "CSVReader"
    )

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    # get column variables
    func_text = "def csv_impl(fname, nrows, skiprows):\n"
    func_text += "    (table_val, idx_col) = _csv_reader_py(fname, nrows, skiprows)\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    csv_impl = loc_vars["csv_impl"]

    # Use the out_used_cols information to determine the final columns
    # to actualy load. For example, if we have the code.
    #
    # T = read_csv(table(0, 1, 2, 3), usecols=[1, 2])
    # arr = T[1]
    #
    # Then after optimizations:
    # usecols = [1, 2]
    # out_used_cols = [1]
    #
    # This computes the columns to actually load based on the offsets:
    # final_usecols = [2]
    #
    # See 'csv_remove_dead_column' for more information.
    #

    # usecols is empty in the case that the table is dead, but not the index.
    # see 'remove_dead_csv' for more information.
    final_usecols = csv_node.usecols
    if final_usecols:
        final_usecols = [csv_node.usecols[i] for i in csv_node.out_used_cols]
    # Add debug info about column pruning
    if bodo.user_logging.get_verbose_level() >= 1:
        msg = "Finish column pruning on read_csv node:\n%s\nColumns loaded %s\n"
        csv_source = csv_node.loc.strformat()
        csv_cols = []
        dict_encoded_cols = []
        if final_usecols:
            # We use csv_node.out_used_cols because this is the actual
            # offset into the type.
            for i in csv_node.out_used_cols:
                colname = csv_node.out_table_col_names[i]
                csv_cols.append(colname)
                if isinstance(
                    csv_node.out_table_col_types[i],
                    bodo.libs.dict_arr_ext.DictionaryArrayType,
                ):
                    dict_encoded_cols.append(colname)
        bodo.user_logging.log_message("Column Pruning", msg, csv_source, csv_cols)
        # TODO: Test. Dictionary encoding isn't supported yet.
        if dict_encoded_cols:
            encoding_msg = "Finished optimized encoding on read_csv node:\n%s\nColumns %s using dictionary encoding to reduce memory usage.\n"
            bodo.user_logging.log_message(
                "Dictionary Encoding",
                encoding_msg,
                csv_source,
                dict_encoded_cols,
            )

    csv_reader_py = _gen_csv_reader_py(
        csv_node.out_table_col_names,
        csv_node.out_table_col_types,
        final_usecols,
        csv_node.out_used_cols,
        csv_node.sep,
        parallel,
        csv_node.header,
        csv_node.compression,
        csv_node.is_skiprows_list,
        csv_node.pd_low_memory,
        csv_node.escapechar,
        csv_node.storage_options,
        idx_col_index=csv_node.index_column_index,
        idx_col_typ=csv_node.index_column_typ,
    )
    f_block = compile_to_numba_ir(
        csv_impl,
        {"_csv_reader_py": csv_reader_py},
        typingctx=typingctx,
        targetctx=targetctx,
        # file_name, nrows, skiprows
        arg_typs=(string_type, types.int64, skiprows_typ),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(
        f_block,
        [
            csv_node.file_name,
            csv_node.nrows,
            csv_node.skiprows,
            csv_node.is_skiprows_list,
        ],
    )
    nodes = f_block.body[:-3]

    # The nodes IR should look somthing like
    # arr0.149 = $12unpack_sequence.5.147
    # idx_col.150 = $12unpack_sequence.6.148
    # and the args are passed in as [table_var, idx_var]
    # Set the lhs of the final two assigns to the passed in variables.
    nodes[-1].target = csv_node.out_vars[1]
    nodes[-2].target = csv_node.out_vars[0]
    # At most one of the table and the index
    # can be dead because otherwise the whole
    # node should have already been removed.
    assert not (csv_node.index_column_index is None and not final_usecols), (
        "At most one of table and index should be dead if the CSV IR node is live"
    )
    if csv_node.index_column_index is None:
        # If the index_col is dead, remove the node.
        nodes.pop(-1)
    elif not final_usecols:
        # If the table is dead, remove the node
        nodes.pop(-2)
    return nodes


def csv_remove_dead_column(csv_node, column_live_map, equiv_vars, typemap):
    """
    Function that tracks which columns to prune from the CSV node.
    This updates out_used_cols which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the actual file columns in 'csv_distributed_run'.
    """
    if csv_node.chunksize is not None:
        # We skip column pruning with chunksize.
        return False
    return bodo.ir.connector.base_connector_remove_dead_columns(
        csv_node,
        column_live_map,
        equiv_vars,
        typemap,
        "CSVReader",
        # usecols is set to an empty list if the table is dead
        # see 'remove_dead_csv'
        csv_node.usecols,
    )


numba.parfors.array_analysis.array_analysis_extensions[CsvReader] = (
    bodo.ir.connector.connector_array_analysis
)
distributed_analysis.distributed_analysis_extensions[CsvReader] = (
    bodo.ir.connector.connector_distributed_analysis
)
typeinfer.typeinfer_extensions[CsvReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[CsvReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[CsvReader] = remove_dead_csv
numba.core.analysis.ir_extension_usedefs[CsvReader] = (
    bodo.ir.connector.connector_usedefs
)
ir_utils.copy_propagate_extensions[CsvReader] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[CsvReader] = (
    bodo.ir.connector.apply_copies_connector
)
ir_utils.build_defs_extensions[CsvReader] = (
    bodo.ir.connector.build_connector_definitions
)
distributed_pass.distributed_run_extensions[CsvReader] = csv_distributed_run
remove_dead_column_extensions[CsvReader] = csv_remove_dead_column
ir_extension_table_column_use[CsvReader] = bodo.ir.connector.connector_table_column_use


def _get_dtype_str(t):
    dtype = t.dtype

    if isinstance(dtype, PDCategoricalDtype):
        cat_arr = CategoricalArrayType(dtype)
        # HACK: add cat type to numba.core.types
        # FIXME: fix after Numba #3372 is resolved
        cat_arr_name = "CategoricalArrayType" + str(ir_utils.next_label())
        setattr(types, cat_arr_name, cat_arr)
        return cat_arr_name

    if dtype == types.NPDatetime("ns"):
        dtype = 'NPDatetime("ns")'

    if t == string_array_type:
        # HACK: add string_array_type to numba.core.types
        # FIXME: fix after Numba #3372 is resolved
        types.string_array_type = string_array_type  # type: ignore
        return "string_array_type"

    if isinstance(t, IntegerArrayType):
        # HACK: same issue as above
        t_name = f"int_arr_{dtype}"
        setattr(types, t_name, t)
        return t_name

    if isinstance(t, FloatingArrayType):
        # HACK: same issue as above
        t_name = f"float_arr_{dtype}"
        setattr(types, t_name, t)
        return t_name

    if t == boolean_array_type:
        types.boolean_array_type = boolean_array_type  # type: ignore
        return "boolean_array_type"

    if dtype == types.bool_:
        dtype = "bool_"

    if dtype == datetime_date_type:
        return "datetime_date_array_type"

    if isinstance(t, ArrayItemArrayType) and isinstance(
        dtype, (StringArrayType, ArrayItemArrayType)
    ):
        # HACK add list of string and nested list type to numba.core.types for objmode
        typ_name = f"ArrayItemArrayType{str(ir_utils.next_label())}"
        setattr(types, typ_name, t)
        return typ_name

    return f"{dtype}[::1]"


def _get_pd_dtype_str(t):
    """Get data type string to pass to df.astype() for Bodo array type

    Args:
        t (types.Type): Bodo array type

    Returns:
        str: data type string (e.g. 'np.int64', 'Int64', ...)
    """
    dtype = t.dtype

    if isinstance(dtype, PDCategoricalDtype):
        return f"pd.CategoricalDtype({dtype.categories})"

    if dtype == types.NPDatetime("ns"):
        return "str"

    # NOTE: this is just a placeholder since strings are not handled with astype()
    if t == string_array_type or t == bodo.types.dict_str_arr_type:
        return "str"

    # nullable int array
    if isinstance(t, IntegerArrayType):
        return '"{}Int{}"'.format("" if dtype.signed else "U", dtype.bitwidth)

    # nullable float
    if isinstance(t, FloatingArrayType):
        # Float32 or Float64
        return f'"{t.get_pandas_scalar_type_instance.name}"'

    if t == boolean_array_type or t == types.Array(types.bool_, 1, "C"):
        return "np.bool_"

    if isinstance(t, ArrayItemArrayType) and isinstance(
        dtype, (StringArrayType, ArrayItemArrayType)
    ):
        return "object"

    return f"np.{dtype}"


# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


@numba.njit(cache=True)
def check_nrows_skiprows_value(nrows, skiprows):
    """Check at runtime that nrows and skiprows values are >= 0"""
    # Corner case: if user did nrows=-1, this will pass. -1 to mean all rows.
    if nrows < -1:
        raise ValueError("pd.read_csv: nrows must be integer >= 0.")
    if skiprows[0] < 0:
        raise ValueError("pd.read_csv: skiprows must be integer >= 0.")


def astype(df, typemap, parallel):
    """Casts the DataFrame read by pd.read_csv to the specified output types.
    The parallel flag determines if errors need to be gathered on all ranks.
    This function is called from inside objmode."""
    message = ""
    from collections import defaultdict

    set_map = defaultdict(list)
    for col_name, col_type in typemap.items():
        set_map[col_type].append(col_name)
    original_columns = df.columns.to_list()
    df_list = []
    for col_type, columns in set_map.items():
        try:
            df_list.append(df.loc[:, columns].astype(col_type, copy=False))
            df = df.drop(columns, axis=1)
        except (ValueError, TypeError) as e:
            message = (
                f"Caught the runtime error '{e}' on columns {columns}."
                " Consider setting the 'dtype' argument in 'read_csv' or investigate"
                " if the data is corrupted."
            )
            break
    raise_error = bool(message)
    if parallel:
        comm = MPI.COMM_WORLD
        raise_error = comm.allreduce(raise_error, op=MPI.LOR)
    if raise_error:
        common_err_msg = "pd.read_csv(): Bodo could not infer dtypes correctly."
        if message:
            raise TypeError(f"{common_err_msg}\n{message}")
        else:
            raise TypeError(f"{common_err_msg}\nPlease refer to errors on other ranks.")
    df = pd.concat(df_list + [df], axis=1)
    result = df.loc[:, original_columns]
    return result


def _gen_csv_file_reader_init(
    parallel,
    header,
    compression,
    chunksize,
    is_skiprows_list,
    pd_low_memory,
    storage_options,
):
    """
    This function generates the f_reader used by pd.read_csv. This f_reader
    may be used for a single pd.read_csv call or a csv_reader used inside
    the csv_iterator.
    """

    # here, header can either be:
    #  0 meaning the first row of the file(s) is the header row
    #  None meaning the file(s) does not contain header
    has_header = header == 0
    # With Arrow 2.0.0, gzip and bz2 map to gzip and bz2 directly
    # and not GZIP and BZ2 like they used to.
    if compression is None:
        compression = "uncompressed"  # Arrow's representation

    # Generate the body to create the file chunk reader. This is shared by the iterator and non iterator
    # implementations.
    # If skiprows is a single value wrap it as a list
    # and pass flag to identify whether skiprows is a list or a single element.
    # This is needed because behavior of skiprows=4 is different from skiprows=[4]
    # and C++ code implementation differs for both cases.
    # The former means skip 4 rows from the beginning. Later means skip the 4th row.
    if is_skiprows_list:
        # TODO: Fix sorted. This line takes ~2 seconds to compile because of how
        # the list sort function is generated.
        func_text = "  skiprows = sorted(set(skiprows))\n"
    else:
        func_text = "  skiprows = [skiprows]\n"
    func_text += "  skiprows_list_len = len(skiprows)\n"
    func_text += "  check_nrows_skiprows_value(nrows, skiprows)\n"
    # check_java_installation is a check for hdfs that java is installed
    func_text += "  check_java_installation(fname)\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"  bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(fname, parallel={parallel})\n"
    # Add a dummy variable to the dict (empty dicts are not yet supported in numba).
    if storage_options is None:
        storage_options = {}
    storage_options["bodo_dummy"] = "dummy"
    func_text += (
        f"  storage_options_py = get_storage_options_pyobject({str(storage_options)})\n"
    )
    func_text += "  f_reader = bodo.ir.csv_ext.csv_file_chunk_reader(bodo.libs.str_ext.unicode_to_utf8(fname), "
    # change skiprows to array
    # pass how many elements in the list as well or 0 if just an integer not a list
    func_text += f"    {parallel}, bodo.utils.conversion.coerce_to_ndarray(skiprows, scalar_to_arr_len=1).ctypes, nrows, {has_header}, bodo.libs.str_ext.unicode_to_utf8('{compression}'), bodo.libs.str_ext.unicode_to_utf8(bucket_region), storage_options_py, {chunksize}, {is_skiprows_list}, skiprows_list_len, {pd_low_memory})\n"
    # TODO: unrelated to skiprows list PR
    # This line is printed even if failure is because of another check
    # Commenting it gives another compiler error.
    # TypeError: csv_reader_py expected 1 argument, got 0
    func_text += "  if bodo.utils.utils.is_null_pointer(f_reader._pyobj):\n"
    func_text += "      raise FileNotFoundError('File does not exist')\n"
    return func_text


def _gen_read_csv_objmode(
    col_names,
    sanitized_cnames,
    col_typs,
    usecols,
    out_used_cols,
    sep,
    escapechar,
    storage_options,
    call_id,
    glbs,
    parallel,
    check_parallel_runtime,
    idx_col_index,
    idx_col_typ,
):
    """
    Generate a code body that calls into objmode to perform read_csv using
    the various function parameters. After read_csv finishes, we cast the
    inferred types to the provided column types, whose implementation
    depends on if the csv read is parallel or sequential.

    This code is shared by both the main csv node and the csv iterator implementation,
    but those differ in how parallel can be determined. Since the csv iterator
    will need to generate this code with a different infrastructure, the parallel vs
    sequential check must be done at runtime. Setting check_parallel_runtime=True will
    ignore the parallel flag and instead use the parallel value stored inside the f_reader
    object (and return it from objmode).

    """

    # Pandas' `read_csv` and Bodo's `read_csv` are not exactly equivalent,
    # for instance in a column of `int64` if there is a missing entry,
    # pandas would convert it to a `float64` column whereas Bodo would use a
    # `Int64` type (nullable integers), etc. We discovered a performance bug with
    # certain nullable types, notably `Int64`, in read_csv, i.e. when we
    # specify the `Int64` dtype in `pd.read_csv`, the performance is very poor.
    # Interestingly, if we do `pd.read_csv` without `dtype` argument and then
    # simply do `df.astype` right after, we do not face the performance
    # penalty. However, when reading strings, if we have missing entries,
    # doing `df.astype` would convert those entries to literally the
    # string values "nan". This is not desirable. Ideally we would use the
    # nullable string type ("string") which would not have this issue, but
    # unfortunately the performance is slow (in both `pd.read_csv` and `df.astype`).
    # Therefore, we have the workaround below where we specify the `dtype` for strings
    # (`str`) directly in `pd.read_csv` (there's no performance penalty, we checked),
    # and specify the rest of the dtypes in the `df.astype` call.

    # NOTE: after optimization
    # usecols refers to the global column indices of the columns being selected whereas
    # out_used_cols refers to the index into the col_names/col_types being chosen.
    # this should be column position in usecols list not w.r.t. original columns
    date_inds_strs = [
        str(i)
        for i, col_num in enumerate(usecols)
        if col_typs[out_used_cols[i]].dtype == types.NPDatetime("ns")
    ]

    # add idx col if needed
    if idx_col_typ == types.NPDatetime("ns"):
        assert idx_col_index is not None
        date_inds_strs.append(str(idx_col_index))

    date_inds = ", ".join(date_inds_strs)

    # _gen_read_csv_objmode() may be called from iternext_impl when
    # used to generate a csv_iterator. That function doesn't have access
    # to the parallel flag in CSVNode so we retrieve it from the file reader.
    parallel_varname = _gen_parallel_flag_name(sanitized_cnames)
    par_var_typ_str = f"{parallel_varname}='bool_'" if check_parallel_runtime else ""

    # array of column numbers that should be specified as str in pd.read_csv()
    # using a global array (constant lowered) for faster compilation for many columns

    # get column type from original column type list
    # out_used_cols includes index of column(s) used from col_typs list
    # col_typs contains datatype of each column in the df
    # use out_used_cols to find which column from usecols is actually used
    # then, use that to index into column types and get original datatype
    usecol_pd_dtypes = [
        _get_pd_dtype_str(col_typs[out_used_cols[i]]) for i in range(len(usecols))
    ]
    index_pd_dtype = None if idx_col_index is None else _get_pd_dtype_str(idx_col_typ)

    str_col_nums_list = [
        col_num for i, col_num in enumerate(usecols) if usecol_pd_dtypes[i] == "str"
    ]

    # add idx col if needed
    if idx_col_index is not None and index_pd_dtype == "str":
        str_col_nums_list.append(idx_col_index)

    str_col_nums = np.array(str_col_nums_list, dtype=np.int64)

    glbs[f"str_col_nums_{call_id}"] = str_col_nums
    # NOTE: assigning a new variable to make globals used inside objmode local to the
    # function, which avoids objmode caching errors
    func_text = f"  str_col_nums_{call_id}_2 = str_col_nums_{call_id}\n"

    # array of used columns to load from pd.read_csv()
    # using a global array (constant lowered) for faster compilation for many columns
    use_cols_arr = np.array(
        usecols + ([idx_col_index] if idx_col_index is not None else []), dtype=np.int64
    )
    glbs[f"usecols_arr_{call_id}"] = use_cols_arr
    func_text += f"  usecols_arr_{call_id}_2 = usecols_arr_{call_id}\n"
    # Array of offsets within the type used for creating the table.
    usecol_type_offset_arr = np.array(out_used_cols, dtype=np.int64)
    if usecols:
        glbs[f"type_usecols_offsets_arr_{call_id}"] = usecol_type_offset_arr
        func_text += f"  type_usecols_offsets_arr_{call_id}_2 = type_usecols_offsets_arr_{call_id}\n"

    # dtypes to specify in the `df.astype` call done right after the `pd.read_csv` call
    # using global arrays (constant lowered) for each type to avoid
    # generating a lot of code (faster compilation for many columns)
    typ_map = defaultdict(list)
    for i, col_num in enumerate(usecols):
        if usecol_pd_dtypes[i] == "str":
            continue
        typ_map[usecol_pd_dtypes[i]].append(col_num)

    # add idx col if needed
    if idx_col_index is not None and index_pd_dtype != "str":
        typ_map[index_pd_dtype].append(idx_col_index)

    for i, t_list in enumerate(typ_map.values()):
        glbs[f"t_arr_{i}_{call_id}"] = np.asarray(t_list)
        func_text += f"  t_arr_{i}_{call_id}_2 = t_arr_{i}_{call_id}\n"

    if idx_col_index != None:
        # idx_array_typ is added to the globals at a higher level
        func_text += f"  with bodo.ir.object_mode.no_warning_objmode(T=table_type_{call_id}, idx_arr=idx_array_typ, {par_var_typ_str}):\n"
    else:
        func_text += f"  with bodo.ir.object_mode.no_warning_objmode(T=table_type_{call_id}, {par_var_typ_str}):\n"
    # create typemap for `df.astype` in runtime
    func_text += "    typemap = {}\n"
    for i, t_str in enumerate(typ_map.keys()):
        func_text += (
            f"    typemap.update({{i:{t_str} for i in t_arr_{i}_{call_id}_2}})\n"
        )
    func_text += "    if f_reader.get_chunk_size() == 0:\n"
    # Pass str as default dtype. Non-str column types will be
    # assigned with `astype` below.
    func_text += (
        f"      df = pd.DataFrame(columns=usecols_arr_{call_id}_2, dtype=str)\n"
    )
    func_text += "    else:\n"
    # Add extra indent for the read_csv call
    func_text += "      df = pd.read_csv(f_reader,\n"
    # header is always None here because header information was found in untyped pass.
    # this pd.read_csv() happens at runtime and is passing a file reader(f_reader)
    # to pandas. f_reader skips the header, so we have to tell pandas header=None.
    func_text += "        header=None,\n"
    func_text += f"        parse_dates=[{date_inds}],\n"
    # Check explanation near top of the function for why we specify
    # only some types here directly
    # NOTE: this works for dict-encoded string arrays too since Bodo's unboxing calls
    # dictionary_encode() if necessary
    func_text += (
        f"        dtype={{i:'string[pyarrow]' for i in str_col_nums_{call_id}_2}},\n"
    )
    # NOTE: using repr() for sep to support cases like "\n" properly
    # and escapechar to support `\\` properly.
    func_text += f"        usecols=[int(i) for i in usecols_arr_{call_id}_2], sep={sep!r}, low_memory=False, escapechar={escapechar!r})\n"
    # _gen_read_csv_objmode() may be called from iternext_impl which doesn't
    # have access to the parallel flag in the CSVNode so we retrieve it from
    # the file reader.
    if check_parallel_runtime:
        func_text += f"    {parallel_varname} = f_reader.is_parallel()\n"
    else:
        func_text += f"    {parallel_varname} = {parallel}\n"
    # Check explanation near top of the function for why we specify
    # some types here rather than directly in the `pd.read_csv` call.
    func_text += f"    df = astype(df, typemap, {parallel_varname})\n"
    # TODO: update and test with usecols
    if idx_col_index != None:
        idx_col_output_index = sorted(use_cols_arr).index(idx_col_index)
        func_text += f"    idx_arr = df.iloc[:, {idx_col_output_index}].values\n"
        func_text += (
            f"    df.drop(columns=df.columns[{idx_col_output_index}], inplace=True)\n"
        )
    # if usecols is empty, the table is dead, see remove_dead_csv.
    # In this case, we simply return None
    if len(usecols) == 0:
        func_text += "    T = None\n"
    else:
        func_text += "    arrs = []\n"
        func_text += "    for i in range(df.shape[1]):\n"
        func_text += "      arr = df.iloc[:, i].values\n"
        func_text += "      arrs.append(arr.astype('datetime64[ns]') if arr.dtype == 'datetime64[us]' else arr)\n"
        # Bodo preserves all of the original types needed at typing in col_typs
        func_text += f"    T = Table(arrs, type_usecols_offsets_arr_{call_id}_2, {len(col_names)})\n"
    return func_text


def _gen_parallel_flag_name(sanitized_cnames):
    """
    Get a unique variable name not found in the
    columns for the parallel flag. This is done
    because the csv_iterator case requires returning
    the value from objmode.
    """
    parallel_varname = "_parallel_value"
    while parallel_varname in sanitized_cnames:
        parallel_varname = "_" + parallel_varname
    return parallel_varname


def _gen_csv_reader_py(
    col_names,
    col_typs,
    usecols,
    out_used_cols,
    sep,
    parallel,
    header,
    compression,
    is_skiprows_list,
    pd_low_memory,
    escapechar,
    storage_options,
    idx_col_index=None,
    idx_col_typ=types.none,
):
    """
    Function that generates the body for a csv_node when chunksize
    is not provided (just read a csv). It creates a function that creates
    a file reader in C++, then calls into pandas to read the csv, and finally
    returns the relevant columns.
    """
    # TODO: support non-numpy types like strings
    sanitized_cnames = [sanitize_varname(c) for c in col_names]
    call_id = create_arg_hash(
        parallel,
        header,
        compression,
        -1,
        is_skiprows_list,
        pd_low_memory,
        storage_options,
        col_names,
        sanitized_cnames,
        col_typs,
        usecols,
        out_used_cols,
        sep,
        escapechar,
        False,
        idx_col_index,
        idx_col_typ,
    )
    func_text = "def bodo_csv_reader_py(fname, nrows, skiprows):\n"
    # If we reached this code path we don't have a chunksize, so set it to -1
    func_text += _gen_csv_file_reader_init(
        parallel,
        header,
        compression,
        -1,
        is_skiprows_list,
        pd_low_memory,
        storage_options,
    )
    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    # {'objmode': objmode, 'csv_file_chunk_reader': csv_file_chunk_reader,
    # 'pd': pd, 'np': np}
    # objmode type variable used in _gen_read_csv_objmode
    if idx_col_typ != types.none:
        glbls["idx_array_typ"] = idx_col_typ

    # in the case that usecols is empty, the table is dead.
    # in this case, we simply return the
    if len(usecols) == 0:
        glbls[f"table_type_{call_id}"] = types.none
    else:
        glbls[f"table_type_{call_id}"] = TableType(tuple(col_typs))
    func_text += _gen_read_csv_objmode(
        col_names,
        sanitized_cnames,
        col_typs,
        usecols,
        out_used_cols,
        sep,
        escapechar,
        storage_options,
        call_id,
        glbls,
        parallel=parallel,
        check_parallel_runtime=False,
        idx_col_index=idx_col_index,
        idx_col_typ=idx_col_typ,
    )
    if idx_col_index != None:
        func_text += "  return (T, idx_arr)\n"
    else:
        func_text += "  return (T, None)\n"
    loc_vars = {}
    glbls["get_storage_options_pyobject"] = get_storage_options_pyobject
    exec(func_text, glbls, loc_vars)
    csv_reader_py = loc_vars["bodo_csv_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    # TODO: objmode and caching doesn't work
    jit_func = numba.njit(csv_reader_py, cache=False)
    compiled_funcs.append(jit_func)

    return jit_func
