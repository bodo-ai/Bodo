# Copyright (C) 2019 Bodo Inc. All rights reserved.

from collections import defaultdict

import numba
import numpy as np  # noqa
import pandas as pd  # noqa
from mpi4py import MPI
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes

import bodo
import bodo.ir.connector
from bodo import objmode  # noqa
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.hiframes.table import Table, TableType  # noqa
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import StringArrayType, string_array_type
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.table_column_del_pass import (
    get_live_column_nums_block,
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import check_java_installation  # noqa
from bodo.utils.utils import sanitize_varname


class CsvReader(ir.Stmt):
    def __init__(
        self,
        file_name,
        df_out,
        sep,
        df_colnames,
        out_vars,
        out_types,
        usecols,
        loc,
        header,
        compression,
        nrows,
        skiprows,
        chunksize,
        is_skiprows_list,
        low_memory,
        index_column_index=None,
        index_column_typ=types.none,
    ):
        self.connector_typ = "csv"
        self.file_name = file_name
        self.df_out = df_out  # used only for printing
        self.sep = sep
        self.df_colnames = df_colnames
        self.out_vars = out_vars
        self.out_types = out_types
        self.usecols = usecols
        self.loc = loc
        self.skiprows = skiprows
        self.nrows = nrows
        self.header = header
        self.compression = compression
        # If this value is not None, we return an iterator instead of a DataFrame.
        # When this happens the out_vars are a list with a single CSVReaderType.
        self.chunksize = chunksize
        # skiprows list
        self.is_skiprows_list = is_skiprows_list
        self.pd_low_memory = low_memory
        self.index_column_index = index_column_index
        self.index_column_typ = index_column_typ
        # Columns within the output table type that are actually used.
        # These will be updated during optimzations and do not contain
        # the actual columns numbers that should be loaded. For more
        # information see 'csv_remove_dead_column'.
        self.type_usecol_offset = list(range(len(usecols)))

    def __repr__(self):  # pragma: no cover
        return "{} = ReadCsv(file={}, col_names={}, types={}, vars={}, nrows={}, skiprows={}, chunksize={}, is_skiprows_list={}, pd_low_memory={}, index_column_index={}, index_colum_typ = {}, type_usecol_offsets={})".format(
            self.df_out,
            self.file_name,
            self.df_colnames,
            self.out_types,
            self.out_vars,
            self.nrows,
            self.skiprows,
            self.chunksize,
            self.is_skiprows_list,
            self.pd_low_memory,
            self.index_column_index,
            self.index_column_typ,
            self.type_usecol_offset,
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
                f"pd.read_csv(): 'skiprows' callable not supported yet.",
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

from bodo.io import csv_cpp

ll.add_symbol("csv_file_chunk_reader", csv_cpp.csv_file_chunk_reader)

csv_file_chunk_reader = types.ExternalFunction(
    "csv_file_chunk_reader",
    bodo.ir.connector.stream_reader_type(
        types.voidptr,
        types.bool_,
        types.voidptr,  # skiprows (array of int64_t)
        types.int64,
        types.bool_,
        types.voidptr,
        types.voidptr,
        types.int64,  # chunksize
        types.bool_,  # is_skiprows_list
        types.int64,  # skiprows_list_len
        types.bool_,  # pd_low_memory
    ),
)


def remove_dead_csv(
    csv_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
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
            csv_node.out_types = []
            csv_node.type_usecol_offset = []

    return csv_node


def csv_distributed_run(
    csv_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    """
    Generate that actual code for this ReadCSV Node during distributed pass.
    This produces different code depending on if the read_csv call contains
    chunksize or not.
    """
    # parallel read flag
    parallel = False
    # skiprows as `ir.Const` indicates default value.
    # If it's a list, it will never be `ir.Const`
    skiprows_typ = (
        types.int64
        if isinstance(csv_node.skiprows, ir.Const)
        else typemap[csv_node.skiprows.name]
    )
    if csv_node.chunksize is not None:
        if array_dists is not None:
            # Parallel flag for iterator is based on the single var.
            iterator_varname = csv_node.out_vars[0].name
            parallel = array_dists[iterator_varname] in (
                distributed_pass.Distribution.OneD,
                distributed_pass.Distribution.OneD_Var,
            )

        # Iterator Case

        # Create a wrapper function that will be compiled. This will return
        # an iterator.
        func_text = "def csv_iterator_impl(fname, nrows, skiprows):\n"
        func_text += f"    reader = _csv_reader_init(fname, nrows, skiprows)\n"
        func_text += f"    iterator = init_csv_iterator(reader, csv_iterator_type)\n"
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
        )
        init_func_text += "  return f_reader\n"
        exec(init_func_text, globals(), loc_vars)
        csv_reader_init = loc_vars["csv_reader_init"]

        # njit the function so it can be called by our outer function.
        # We keep track of the function for possible dynamic addresses
        jit_func = numba.njit(csv_reader_init)
        compiled_funcs.append(jit_func)
        skiprows_typ = (
            types.int64
            if isinstance(csv_node.skiprows, ir.Const)
            else typemap[csv_node.skiprows.name]
        )

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
    if array_dists is not None:
        # table is parallel
        table_varname = csv_node.out_vars[0].name
        parallel = array_dists[table_varname] in (
            distributed_pass.Distribution.OneD,
            distributed_pass.Distribution.OneD_Var,
        )
        index_varname = csv_node.out_vars[1].name
        # index array parallelism should match the table
        assert (
            typemap[index_varname] == types.none
            or not parallel
            or array_dists[index_varname]
            in (
                distributed_pass.Distribution.OneD,
                distributed_pass.Distribution.OneD_Var,
            )
        ), "pq data/index parallelization does not match"

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    # get column variables
    func_text = "def csv_impl(fname, nrows, skiprows):\n"
    func_text += f"    (table_val, idx_col) = _csv_reader_py(fname, nrows, skiprows)\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    csv_impl = loc_vars["csv_impl"]

    # Use the type_usecol_offset information to determine the final columns
    # to actualy load. For example, if we have the code.
    #
    # T = read_csv(table(0, 1, 2, 3), usecols=[1, 2])
    # arr = T[1]
    #
    # Then after optimizations:
    # usecols = [1, 2]
    # type_usecol_offset = [1]
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
        final_usecols = [csv_node.usecols[i] for i in csv_node.type_usecol_offset]
    csv_reader_py = _gen_csv_reader_py(
        csv_node.df_colnames,
        csv_node.out_types,
        final_usecols,
        csv_node.type_usecol_offset,
        csv_node.sep,
        parallel,
        csv_node.header,
        csv_node.compression,
        csv_node.is_skiprows_list,
        csv_node.pd_low_memory,
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
    This updates type_usecol_offset which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the actual file columns in 'csv_distributed_run'.
    """
    if csv_node.chunksize is not None:
        # We skip column pruning with chunksize.
        return False
    # All csv_nodes should have a two variables, the first being the table, and the second being the idxcol
    assert len(csv_node.out_vars) == 2, "invalid CsvReader node"
    table_var_name = csv_node.out_vars[0].name
    # out_vars[0] is either a TableType (the normal case) or a reader
    # (the chunksize case). In latter case we don't eliminate any columns
    # at the source (consistent with the previous DataFrame type).
    # if csv_node.usecols is empty, the table is dead. See remove_dead_csv
    if isinstance(typemap[table_var_name], TableType) and csv_node.usecols:
        # Compute all columns that are live at this statement.
        used_columns, use_all = get_live_column_nums_block(
            column_live_map, equiv_vars, table_var_name
        )
        used_columns = bodo.ir.connector.trim_extra_used_columns(
            used_columns, len(csv_node.usecols)
        )
        if not use_all and not used_columns:
            # If we see no specific column is need some operations need some
            # column but no specific column. For example:
            # T = read_csv(table(0, 1, 2, 3))
            # n = len(T)
            #
            # Here we just load column 0. If no columns are actually needed, dead
            # code elimination will remove the entire IR var in 'remove_dead_csv'.
            #
            used_columns = [0]
        if not use_all and len(used_columns) != len(csv_node.type_usecol_offset):
            # Update the type offset. If we have code like
            #
            # T = read_csv(table(0, 1, 2, 3), usecols=[1, 2])
            #
            # Then T is typed after applying use cols, so the type
            # is Table(arr1, arr2). As a result once we apply optimizations,
            # all the column indices will refer to the index within that
            # type, not the original file.
            #
            # i.e. T[1] == arr2
            #
            # This means that used_columns will track the offsets within the type,
            # not the actual column numbers in the file. We keep these offsets separate
            # while finalizing DCE and we will update the file with the actual columns later
            # in 'csv_distributed_run'.
            #
            # For more information see:
            # https://bodo.atlassian.net/wiki/spaces/B/pages/921042953/Table+Structure+with+Dead+Columns#User-Provided-Column-Pruning-at-the-Source

            csv_node.type_usecol_offset = used_columns
            # Return that this table was updated
            return True
    return False


def csv_table_column_use(csv_node, block_use_map, equiv_vars, typemap):
    """
    Function to handle any necessary processing for column uses
    with a particular table. CsvReader defines a table and doesn't
    use any other table, so this does nothing.
    """
    return


numba.parfors.array_analysis.array_analysis_extensions[
    CsvReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    CsvReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[CsvReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[CsvReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[CsvReader] = remove_dead_csv
numba.core.analysis.ir_extension_usedefs[
    CsvReader
] = bodo.ir.connector.connector_usedefs
ir_utils.copy_propagate_extensions[CsvReader] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[
    CsvReader
] = bodo.ir.connector.apply_copies_connector
ir_utils.build_defs_extensions[
    CsvReader
] = bodo.ir.connector.build_connector_definitions
distributed_pass.distributed_run_extensions[CsvReader] = csv_distributed_run
remove_dead_column_extensions[CsvReader] = csv_remove_dead_column
ir_extension_table_column_use[CsvReader] = csv_table_column_use


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
        types.string_array_type = string_array_type
        return "string_array_type"

    if isinstance(t, IntegerArrayType):
        # HACK: same issue as above
        t_name = "int_arr_{}".format(dtype)
        setattr(types, t_name, t)
        return t_name

    if t == boolean_array:
        types.boolean_array = boolean_array
        return "boolean_array"

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

    return "{}[::1]".format(dtype)


def _get_pd_dtype_str(t):
    dtype = t.dtype

    if isinstance(dtype, PDCategoricalDtype):
        return "pd.CategoricalDtype({})".format(dtype.categories)

    if dtype == types.NPDatetime("ns"):
        return "str"

    if t == string_array_type:
        return "str"

    # nullable int array
    if isinstance(t, IntegerArrayType):
        return '"{}Int{}"'.format("" if dtype.signed else "U", dtype.bitwidth)

    if t == boolean_array:
        return "np.bool_"

    if isinstance(t, ArrayItemArrayType) and isinstance(
        dtype, (StringArrayType, ArrayItemArrayType)
    ):
        return "object"

    return "np.{}".format(dtype)


# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


@numba.njit
def check_nrows_skiprows_value(nrows, skiprows):
    """ Check at runtime that nrows and skiprows values are >= 0 """
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
        except TypeError as e:
            message = (
                f"Caught the TypeError '{e}' on columns {columns}."
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
        func_text = "  skiprows = sorted(set(skiprows))\n"
    else:
        func_text = "  skiprows = [skiprows]\n"
    func_text += "  skiprows_list_len = len(skiprows)\n"
    func_text += "  check_nrows_skiprows_value(nrows, skiprows)\n"
    # check_java_installation is a check for hdfs that java is installed
    func_text += "  check_java_installation(fname)\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"  bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname, parallel={parallel})\n"
    func_text += "  f_reader = bodo.ir.csv_ext.csv_file_chunk_reader(bodo.libs.str_ext.unicode_to_utf8(fname), "
    # change skiprows to array
    # pass how many elements in the list as well or 0 if just an integer not a list
    func_text += "    {}, bodo.utils.conversion.coerce_to_ndarray(skiprows, scalar_to_arr_len=1).ctypes, nrows, {}, bodo.libs.str_ext.unicode_to_utf8('{}'), bodo.libs.str_ext.unicode_to_utf8(bucket_region), {}, {}, skiprows_list_len, {})\n".format(
        parallel,
        has_header,
        compression,
        chunksize,
        is_skiprows_list,
        pd_low_memory,
    )
    # Check if there was an error in the C++ code. If so, raise it.
    func_text += "  bodo.utils.utils.check_and_propagate_cpp_exception()\n"
    # TODO: unrelated to skiprows list PR
    # This line is printed even if failure is because of another check
    # Commenting it gives another compiler error.
    # TypeError: csv_reader_py expected 1 argument, got 0
    func_text += "  if bodo.utils.utils.is_null_pointer(f_reader):\n"
    func_text += "      raise FileNotFoundError('File does not exist')\n"
    return func_text


def _gen_read_csv_objmode(
    col_names,
    sanitized_cnames,
    col_typs,
    usecols,
    type_usecol_offset,
    sep,
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

    date_inds = ", ".join(
        str(col_num)
        for i, col_num in enumerate(usecols)
        if col_typs[type_usecol_offset[i]].dtype == types.NPDatetime("ns")
    )

    # add idx col if needed
    if idx_col_typ == types.NPDatetime("ns"):
        assert not idx_col_index is None
        date_inds += ", " + str(idx_col_index)

    # _gen_read_csv_objmode() may be called from iternext_impl when
    # used to generate a csv_iterator. That function doesn't have access
    # to the parallel flag in CSVNode so we retrieve it from the file reader.
    parallel_varname = _gen_parallel_flag_name(sanitized_cnames)
    par_var_typ_str = f"{parallel_varname}='bool_'" if check_parallel_runtime else ""

    # array of column numbers that should be specified as str in pd.read_csv()
    # using a global array (constant lowered) for faster compilation for many columns

    str_col_nums_list = [
        col_num
        for i, col_num in enumerate(usecols)
        if _get_pd_dtype_str(col_typs[type_usecol_offset[i]]) == "str"
    ]

    # add idx col if needed
    if idx_col_index is not None and _get_pd_dtype_str(idx_col_typ) == "str":
        str_col_nums_list.append(idx_col_index)

    str_col_nums = np.array(str_col_nums_list)

    glbs[f"str_col_nums_{call_id}"] = str_col_nums
    # NOTE: assigning a new variable to make globals used inside objmode local to the
    # function, which avoids objmode caching errors
    func_text = f"  str_col_nums_{call_id}_2 = str_col_nums_{call_id}\n"

    # array of used columns to load from pd.read_csv()
    # using a global array (constant lowered) for faster compilation for many columns
    use_cols_arr = np.array(
        usecols + ([idx_col_index] if idx_col_index is not None else [])
    )
    glbs[f"usecols_arr_{call_id}"] = use_cols_arr
    func_text += f"  usecols_arr_{call_id}_2 = usecols_arr_{call_id}\n"
    # Array of offsets within the type used for creating the table.
    usecol_type_offset_arr = np.array(type_usecol_offset)
    if usecols:
        glbs[f"type_usecols_offsets_arr_{call_id}"] = usecol_type_offset_arr
        func_text += f"  type_usecols_offsets_arr_{call_id}_2 = type_usecols_offsets_arr_{call_id}\n"

    # dtypes to specify in the `df.astype` call done right after the `pd.read_csv` call
    # using global arrays (constant lowered) for each type to avoid
    # generating a lot of code (faster compilation for many columns)
    typ_map = defaultdict(list)
    for i, col_num in enumerate(usecols):
        t = col_typs[type_usecol_offset[i]]
        if _get_pd_dtype_str(t) == "str":
            continue
        typ_map[_get_pd_dtype_str(t)].append(col_num)

    # add idx col if needed
    if idx_col_index is not None and _get_pd_dtype_str(idx_col_typ) != "str":
        typ_map[_get_pd_dtype_str(idx_col_typ)].append(idx_col_index)

    for i, t_list in enumerate(typ_map.values()):
        glbs[f"t_arr_{i}_{call_id}"] = np.asarray(t_list)
        func_text += f"  t_arr_{i}_{call_id}_2 = t_arr_{i}_{call_id}\n"

    if idx_col_index != None:
        # idx_array_typ is added to the globals at a higher level
        func_text += f"  with objmode(T=table_type_{call_id}, idx_arr=idx_array_typ, {par_var_typ_str}):\n"
    else:
        func_text += f"  with objmode(T=table_type_{call_id}, {par_var_typ_str}):\n"
    # create typemap for `df.astype` in runtime
    func_text += f"    typemap = {{}}\n"
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
    func_text += "        parse_dates=[{}],\n".format(date_inds)
    # Check explanation near top of the function for why we specify
    # only some types here directly
    func_text += f"        dtype={{i:str for i in str_col_nums_{call_id}_2}},\n"
    # NOTE: using repr() for sep to support cases like "\n" properly
    func_text += (
        f"        usecols=usecols_arr_{call_id}_2, sep={sep!r}, low_memory=False)\n"
    )
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
        func_text += f"    T = None\n"
    else:
        func_text += f"    arrs = []\n"
        func_text += f"    for i in range(df.shape[1]):\n"
        func_text += f"      arrs.append(df.iloc[:, i].values)\n"
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
    type_usecol_offset,
    sep,
    parallel,
    header,
    compression,
    is_skiprows_list,
    pd_low_memory,
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
    func_text = "def csv_reader_py(fname, nrows, skiprows):\n"
    # If we reached this code path we don't have a chunksize, so set it to -1
    func_text += _gen_csv_file_reader_init(
        parallel,
        header,
        compression,
        -1,
        is_skiprows_list,
        pd_low_memory,
    )
    # a unique int used to create global variables with unique names
    call_id = ir_utils.next_label()
    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    # {'objmode': objmode, 'csv_file_chunk_reader': csv_file_chunk_reader,
    # 'pd': pd, 'np': np}
    # objmode type variable used in _gen_read_csv_objmode
    if idx_col_typ != types.none:
        glbls[f"idx_array_typ"] = idx_col_typ

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
        type_usecol_offset,
        sep,
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
    exec(func_text, glbls, loc_vars)
    csv_reader_py = loc_vars["csv_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(csv_reader_py)
    compiled_funcs.append(jit_func)

    return jit_func
