"""IR node for the data sorting"""

from collections import defaultdict

import numba
import numpy as np
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    mk_unique_var,
    replace_arg_nodes,
    replace_vars_inner,
    visit_vars_inner,
)

import bodo
from bodo.libs.array import (
    arr_info_list_to_table,
    array_from_cpp_table,
    array_to_info,
    cpp_table_to_py_data,
    delete_table,
    py_data_to_cpp_table,
    sort_table_for_interval_join,
    sort_values_table_py_entry,
)
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import (
    _compute_table_column_uses,
    _find_used_columns,
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.typing import MetaType, is_overload_none, type_has_unknown_cats
from bodo.utils.utils import gen_getitem


class Sort(ir.Stmt):
    def __init__(
        self,
        df_in: str,
        df_out: str,
        in_vars: list[ir.Var],
        out_vars: list[ir.Var],
        key_inds: tuple[int],
        inplace: bool,
        loc: ir.Loc,
        ascending_list: list[bool] | bool = True,
        na_position: list[str] | str = "last",
        _bodo_chunk_bounds: ir.Var | None = None,
        _bodo_interval_sort: bool = False,
        is_table_format: bool = False,
        num_table_arrays: int = 0,
    ):
        """IR node for sort operations. Produced by several sort operations like
        df/Series.sort_values and df/Series.sort_index.
        Sort IR node allows analysis and optimization throughout the compiler pipeline
        such as removing dead columns. The implementation calls Timsort in the C++
        runtime.

        Args:
            df_in (str): name of input dataframe (for printing Sort node only)
            df_out (str): name of output dataframe (for printing Sort node only)
            in_vars (list[ir.Var]): list of input variables to sort (including keys).
                The first variable is the table for the table format case.
            out_vars (list[ir.Var]): list of output variables of sort (including keys).
                The first variable is the table for the table format case.
            key_inds (tuple[int]): indices of key arrays in in_vars for sorting.
                Indices are logical indices of columns that could be in the table
                (in case of table format) or not.
            inplace (bool): sort values inplace (avoid creating new arrays)
            loc (ir.Loc): location object of this IR node
            ascending_list (bool|list[bool], optional): Ascending or descending sort
                order (can be set per key). Defaults to True.
            na_position (str|list[str], optional): Place null values first or last in
                output array. Can be set per key. Defaults to "last".
            _bodo_chunk_bounds (ir.Var|None): parallel chunk bounds for data
                redistribution during the parallel algorithm (optional).
                Currently only used for Iceberg MERGE INTO, window functions without
                partitions (e.g. ROW_NUMBER) and with _bodo_interval_sort.
            _bodo_interval_sort (bool): Use sort_table_for_interval_join instead of regular
                sort_values. This is only exposed for internal testing purposes.
                When this is true, _bodo_chunk_bounds must not be None and be of length
                (#ranks - 1). If number of keys is 1, we treat it as a point column
                and if it's 2, we treat it as an interval where the first key is the start
                of the interval and the second key is the end of the interval.
                `validate_sort_values_spec` ensures that the number of keys is either 1 or 2
                and _bodo_chunk_bounds is not None.
            is_table_format (bool): flag for table format case (first variable is a
                table in in_vars/out_vars)
            num_table_arrays: number of columns in the table in case of table format
        """

        self.df_in = df_in
        self.df_out = df_out
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.key_inds = key_inds
        self.inplace = inplace
        self._bodo_chunk_bounds = _bodo_chunk_bounds
        self._bodo_interval_sort = _bodo_interval_sort
        self.is_table_format = is_table_format
        self.num_table_arrays = num_table_arrays
        # Logical indices of dead columns in input/output (excluding keys), updated in
        # DCE. The column may be in the table (in case of table format) or not.
        # Example: 3 columns table, 1 non-table, dead_var_inds={1, 3} means column 1 in
        # the table is dead and also the non-table column is dead.
        self.dead_var_inds: set[int] = set()
        # Logical indices of dead key columns in output, updated in DCE.
        self.dead_key_var_inds: set[int] = set()

        # normalize na_position to list of bools (per key)
        if isinstance(na_position, str):
            if na_position == "last":
                self.na_position_b = (True,) * len(key_inds)
            else:
                self.na_position_b = (False,) * len(key_inds)
        else:
            self.na_position_b = tuple(
                [
                    True if col_na_position == "last" else False
                    for col_na_position in na_position
                ]
            )

        # normalize ascending to list of bools (per key)
        if isinstance(ascending_list, bool):
            ascending_list = (ascending_list,) * len(key_inds)

        self.ascending_list = ascending_list
        self.loc = loc

    def get_live_in_vars(self):
        """return input variables that are live (handles both table and non-table
        format cases)

        Returns:
            list(ir.Var): list of live variables
        """
        return [v for v in self.in_vars if v is not None]

    def get_live_out_vars(self):
        """return output variables that are live (handles both table and non-table format)

        Returns:
            list(ir.Var): list of live output variables
        """
        return [v for v in self.out_vars if v is not None]

    def __repr__(self):  # pragma: no cover
        in_cols = ", ".join(v.name for v in self.get_live_in_vars())
        df_in_str = f"{self.df_in}{{{in_cols}}}"
        out_cols = ", ".join(v.name for v in self.get_live_out_vars())
        df_out_str = f"{self.df_out}{{{out_cols}}}"
        return f"Sort (keys: {self.key_inds}): {df_in_str} {df_out_str}"


def sort_array_analysis(sort_node, equiv_set, typemap, array_analysis):
    """Array analysis for Sort IR node. Input arrays have the same size. Output arrays
    have the same size as well.
    Inputs and outputs may not have the same local size after shuffling in parallel sort
    so we avoid adding equivalence for them to be conservative (1D_Var handling is
    challenging so the gains don't seem worth it).

    Args:
        sort_node (ir.Sort): input Sort node
        equiv_set (SymbolicEquivSet): equivalence set object of Numba array analysis
        typemap (dict[str, types.Type]): typemap from analysis pass
        array_analysis (ArrayAnalysis): array analysis object for the pass

    Returns:
        tuple(list(ir.Stmt), list(ir.Stmt)): lists of IR statements to add to IR before
        this node and after this node.
    """

    # arrays of input df have same size in first dimension
    all_shapes = []
    for col_var in sort_node.get_live_in_vars():
        col_shape = equiv_set.get_shape(col_var)
        if col_shape is not None:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variables for output columns
    post = []
    all_shapes = []

    for col_var in sort_node.get_live_out_vars():
        typ = typemap[col_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None, post)
        equiv_set.insert_equiv(col_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.parfors.array_analysis.array_analysis_extensions[Sort] = sort_array_analysis


def sort_distributed_analysis(sort_node, array_dists):
    """Distributed analysis for Sort IR node. Inputs and outputs have the same
    distribution, except that output of 1D is 1D_Var due to shuffling.

    Args:
        sort_node (Sort): Sort IR node
        array_dists (dict[str, Distribution]): distributions of arrays in the IR
            (variable name -> Distribution)
    """

    in_arrs = sort_node.get_live_in_vars()
    out_arrs = sort_node.get_live_out_vars()
    # input columns have same distribution
    in_dist = Distribution.OneD
    for col_var in in_arrs:
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))

    # output is 1D_Var due to shuffle, has to meet input dist
    # TODO: set to input dist in inplace case
    out_dist = Distribution(min(in_dist.value, Distribution.OneD_Var.value))
    for col_var in out_arrs:
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value)
            )

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        in_dist = out_dist

    # set dists
    for col_var in in_arrs:
        array_dists[col_var.name] = in_dist

    for col_var in out_arrs:
        array_dists[col_var.name] = out_dist


distributed_analysis.distributed_analysis_extensions[Sort] = sort_distributed_analysis


def sort_typeinfer(sort_node, typeinferer):
    """Type inference extension for Sort IR nodes. Corresponding input and output
    variables have the same type.

    Args:
        sort_node (Sort): Sort IR node
        typeinferer (TypeInferer): type inference pass object
    """

    # input and output arrays have the same type
    for i, out_var in enumerate(sort_node.out_vars):
        in_var = sort_node.in_vars[i]
        if in_var is not None and out_var is not None:
            typeinferer.constraints.append(
                typeinfer.Propagate(
                    dst=out_var.name, src=in_var.name, loc=sort_node.loc
                )
            )


typeinfer.typeinfer_extensions[Sort] = sort_typeinfer


def build_sort_definitions(sort_node, definitions=None):
    """Sort IR node extension for building varibale definitions pass

    Args:
        sort_node (Sort): Sort IR node
        definitions (defaultdict(list), optional): Existing definitions list. Defaults
            to None.

    Returns:
        defaultdict(list): updated definitions
    """

    if definitions is None:
        definitions = defaultdict(list)

    # output arrays are defined
    if not sort_node.inplace:
        for col_var in sort_node.get_live_out_vars():
            definitions[col_var.name].append(sort_node)

    return definitions


ir_utils.build_defs_extensions[Sort] = build_sort_definitions


def visit_vars_sort(sort_node, callback, cbdata):
    """Sort IR node extension for visiting variables pass

    Args:
        sort_node (Sort): Sort IR node
        callback (function): callback to call on each variable (just passed along here)
        cbdata (object): data to pass to callback (just passed along here)
    """

    for i in range(len(sort_node.in_vars)):
        if sort_node.in_vars[i] is not None:
            sort_node.in_vars[i] = visit_vars_inner(
                sort_node.in_vars[i], callback, cbdata
            )
        if sort_node.out_vars[i] is not None:
            sort_node.out_vars[i] = visit_vars_inner(
                sort_node.out_vars[i], callback, cbdata
            )

    if sort_node._bodo_chunk_bounds is not None:
        sort_node._bodo_chunk_bounds = visit_vars_inner(
            sort_node._bodo_chunk_bounds, callback, cbdata
        )


ir_utils.visit_vars_extensions[Sort] = visit_vars_sort


def remove_dead_sort(
    sort_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """Dead code elimination for Sort IR node

    Args:
        sort_node (Sort): Sort IR node
        lives_no_aliases (set(str)): live variable names without their aliases
        lives (set(str)): live variable names with their aliases
        arg_aliases (set(str)): variables that are function arguments or alias them
        alias_map (dict(str, set(str))): mapping of variables names and their aliases
        func_ir (FunctionIR): full function IR
        typemap (dict(str, types.Type)): typemap of variables

    Returns:
        (Sort, optional): Sort IR node if not fully dead, None otherwise
    """

    # TODO: arg aliases for inplace case?

    # table case: get logical indices of dead columns
    if sort_node.is_table_format:
        table_var = sort_node.out_vars[0]
        if table_var is not None and table_var.name not in lives:
            sort_node.out_vars[0] = None
            dead_cols = set(range(sort_node.num_table_arrays))
            key_set = set(sort_node.key_inds)
            sort_node.dead_key_var_inds.update(dead_cols & key_set)
            sort_node.dead_var_inds.update(dead_cols - key_set)
            # input table is dead if no key is from input table
            if len(key_set & dead_cols) == 0:
                sort_node.in_vars[0] = None
        for i in range(1, len(sort_node.out_vars)):
            v = sort_node.out_vars[i]
            if v is not None and v.name not in lives:
                sort_node.out_vars[i] = None
                col_no = sort_node.num_table_arrays + i - 1
                if col_no in sort_node.key_inds:
                    sort_node.dead_key_var_inds.add(col_no)
                else:
                    sort_node.dead_var_inds.add(col_no)
                    sort_node.in_vars[i] = None
    else:
        for i in range(len(sort_node.out_vars)):
            v = sort_node.out_vars[i]
            if v is not None and v.name not in lives:
                sort_node.out_vars[i] = None
                if i in sort_node.key_inds:
                    sort_node.dead_key_var_inds.add(i)
                else:
                    sort_node.dead_var_inds.add(i)
                    sort_node.in_vars[i] = None

    # remove empty sort node
    if all(v is None for v in sort_node.out_vars):
        return None

    return sort_node


ir_utils.remove_dead_extensions[Sort] = remove_dead_sort


def sort_usedefs(sort_node, use_set=None, def_set=None):
    """use/def analysis extension for Sort IR node

    Args:
        sort_node (Sort): Sort IR node
        use_set (set(str), optional): Existing set of used variables. Defaults to None.
        def_set (set(str), optional): Existing set of defined variables. Defaults to
            None.

    Returns:
        namedtuple('use_defs_result', 'usemap,defmap'): use/def sets
    """
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # input arrays are used
    use_set.update({v.name for v in sort_node.get_live_in_vars()})

    # output arrays are defined
    if not sort_node.inplace:
        def_set.update({v.name for v in sort_node.get_live_out_vars()})

    if sort_node._bodo_chunk_bounds is not None:
        use_set.add(sort_node._bodo_chunk_bounds.name)

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Sort] = sort_usedefs


def get_copies_sort(sort_node, typemap):
    """Sort IR node extension for variable copy analysis

    Args:
        sort_node (Sort): Sort IR node
        typemap (dict(str, ir.Var)): typemap of variables

    Returns:
        tuple(set(str), set(str)): set of copies generated or killed
    """
    # sort doesn't generate copies, it just kills the output columns
    kill_set = set()
    if not sort_node.inplace:
        kill_set.update({v.name for v in sort_node.get_live_out_vars()})
    return set(), kill_set


ir_utils.copy_propagate_extensions[Sort] = get_copies_sort


def apply_copies_sort(
    sort_node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """Sort IR node extension for applying variable copies pass

    Args:
        sort_node (Sort): Sort IR node
        var_dict (dict(str, ir.Var)): dictionary of variables to replace
        name_var_table (dict(str, ir.Var)): map variable name to its ir.Var object
        typemap (dict(str, ir.Var)): typemap of variables
        calltypes (dict[ir.Inst, Signature]): signature of callable nodes
        save_copies (list(tuple(str, ir.Var))): copies that were applied
    """
    for i in range(len(sort_node.in_vars)):
        if sort_node.in_vars[i] is not None:
            sort_node.in_vars[i] = replace_vars_inner(sort_node.in_vars[i], var_dict)
        if sort_node.out_vars[i] is not None:
            sort_node.out_vars[i] = replace_vars_inner(sort_node.out_vars[i], var_dict)

    if sort_node._bodo_chunk_bounds is not None:
        sort_node._bodo_chunk_bounds = replace_vars_inner(
            sort_node._bodo_chunk_bounds, var_dict
        )


ir_utils.apply_copy_propagate_extensions[Sort] = apply_copies_sort


def sort_distributed_run(
    sort_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    """lowers Sort IR node to regular IR nodes. Uses the C++ Timsort implementation

    Args:
        sort_node (Sort): Sort IR node to lower
        array_dists (dict(str, Distribution)): distribution of arrays
        typemap (dict(str, ir.Var)): typemap of variables
        calltypes (dict[ir.Inst, Signature]): signature of callable nodes
        typingctx (typing.Context): typing context for compiler pipeline
        targetctx (cpu.CPUContext): target context for compiler pipeline

    Returns:
        list(ir.Stmt): list of IR nodes that implement the input Sort IR node
    """

    parallel = False
    in_vars = sort_node.get_live_in_vars()
    out_vars = sort_node.get_live_out_vars()
    assert len(out_vars) > 0, "Invalid empty Sort node in distributed pass"

    if array_dists is not None:
        parallel = True
        for v in in_vars + out_vars:
            if (
                array_dists[v.name] != distributed_pass.Distribution.OneD
                and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
            ):
                parallel = False

    if not parallel:
        # If we have a replicated call remove sort_node._bodo_chunk_bounds, which
        # is only valid for distributed calls. When dead code elimination runs at
        # the end of distributed pass this should optimize out the _bodo_chunk_bounds
        # variable if it exists.
        sort_node._bodo_chunk_bounds = None

    nodes = []
    out_types = [
        typemap[v.name] if v is not None else types.none for v in sort_node.out_vars
    ]

    func_text, glbls = get_sort_cpp_section(sort_node, out_types, typemap, parallel)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sort_impl = loc_vars["f"]

    glbls.update(
        {
            "bodo": bodo,
            "np": np,
            "delete_table": delete_table,
            "array_from_cpp_table": array_from_cpp_table,
            "sort_values_table_py_entry": sort_values_table_py_entry,
            "sort_table_for_interval_join": sort_table_for_interval_join,
            "arr_info_list_to_table": arr_info_list_to_table,
            "array_to_info": array_to_info,
            "py_data_to_cpp_table": py_data_to_cpp_table,
            "cpp_table_to_py_data": cpp_table_to_py_data,
        }
    )
    glbls.update({f"out_type{i}": out_types[i] for i in range(len(out_types))})

    bounds = sort_node._bodo_chunk_bounds
    bounds_var = bounds
    if bounds is None:
        loc = sort_node.loc
        bounds_var = ir.Var(out_vars[0].scope, mk_unique_var("$bounds_none"), loc)
        typemap[bounds_var.name] = types.none
        nodes.append(ir.Assign(ir.Const(None, loc), bounds_var, loc))

    f_block = compile_to_numba_ir(
        sort_impl,
        glbls,
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=tuple(typemap[v.name] for v in in_vars) + (typemap[bounds_var.name],),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, in_vars + [bounds_var])
    # get return value from cast node, the last node before cast isn't output assignment
    ret_var = f_block.body[-2].value.value
    nodes += f_block.body[:-2]

    for i, v in enumerate(out_vars):
        gen_getitem(v, ret_var, i, calltypes, nodes)

    # TODO: handle 1D balance for inplace case
    return nodes


distributed_pass.distributed_run_extensions[Sort] = sort_distributed_run


def get_sort_cpp_section(sort_node, out_types, typemap, parallel):
    """generate function text for passing arrays to C++, calling sort, and returning
    outputs in correct order.

    Args:
        sort_node (Sort): sort IR node
        parallel (bool): flag for parallel sort

    Returns:
        str: function text for calling C++ sort.
    """

    key_count = len(sort_node.key_inds)
    n_in_vars = len(sort_node.in_vars)
    n_out_vars = len(sort_node.out_vars)
    n_cols = (
        sort_node.num_table_arrays + n_in_vars - 1
        if sort_node.is_table_format
        else n_in_vars
    )

    # get Python/C++ input/output mapping indices
    in_cpp_col_inds, out_cpp_col_inds, dead_keys = _get_cpp_col_ind_mappings(
        sort_node.key_inds,
        sort_node.dead_var_inds,
        sort_node.dead_key_var_inds,
        n_cols,
    )

    # make sure array arg names have logical column number for easier codegen below
    in_args = []
    if sort_node.is_table_format:
        in_args.append("arg0")
        for i in range(1, n_in_vars):
            col_no = sort_node.num_table_arrays + i - 1
            if col_no not in sort_node.dead_var_inds:
                in_args.append(f"arg{col_no}")
    else:
        for i in range(n_cols):
            if i not in sort_node.dead_var_inds:
                in_args.append(f"arg{i}")

    func_text = f"def f({', '.join(in_args)}, bounds_in):\n"

    if sort_node.is_table_format:
        comma = "," if n_in_vars - 1 == 1 else ""
        other_vars = []
        for i in range(sort_node.num_table_arrays, n_cols):
            if i in sort_node.dead_var_inds:
                other_vars.append("None")
            else:
                other_vars.append(f"arg{i}")
        func_text += f"  in_cpp_table = py_data_to_cpp_table(arg0, ({', '.join(other_vars)}{comma}), in_col_inds, {sort_node.num_table_arrays})\n"
    else:
        py_to_cpp_inds = {k: i for i, k in enumerate(in_cpp_col_inds)}
        arr_vars = [None] * len(in_cpp_col_inds)
        for i in range(n_cols):
            out_ind = py_to_cpp_inds.get(i, -1)
            if out_ind != -1:
                arr_vars[out_ind] = f"array_to_info(arg{i})"
        func_text += "  info_list_total = [{}]\n".format(",".join(arr_vars))
        func_text += "  in_cpp_table = arr_info_list_to_table(info_list_total)\n"

    func_text += "  vect_ascending = np.array([{}], np.int64)\n".format(
        ",".join("1" if x else "0" for x in sort_node.ascending_list)
    )
    func_text += "  na_position = np.array([{}], np.int64)\n".format(
        ",".join("1" if x else "0" for x in sort_node.na_position_b)
    )
    func_text += "  dead_keys = np.array([{}], np.int64)\n".format(
        ",".join("1" if i in dead_keys else "0" for i in range(key_count))
    )
    # single-element numpy array to return number of rows from C++
    func_text += "  total_rows_np = np.array([0], dtype=np.int64)\n"
    bounds_in = sort_node._bodo_chunk_bounds
    bounds_table = (
        "0"
        if bounds_in is None or is_overload_none(typemap[bounds_in.name])
        else "arr_info_list_to_table([array_to_info(bounds_in)])"
    )

    # NOTE: C++ will delete in_cpp_table pointer

    if sort_node._bodo_interval_sort:
        # bounds_in must exist if _bodo_interval_sort
        bounds_arr = "array_to_info(bounds_in)"
        func_text += f"  out_cpp_table = sort_table_for_interval_join(in_cpp_table, {bounds_arr}, {bool(key_count == 1)}, {parallel})\n"
    else:
        func_text += f"  out_cpp_table = sort_values_table_py_entry(in_cpp_table, {key_count}, vect_ascending.ctypes, na_position.ctypes, dead_keys.ctypes, total_rows_np.ctypes, {bounds_table}, {parallel})\n"

    if sort_node.is_table_format:
        comma = "," if n_out_vars == 1 else ""
        out_types_str = f"({', '.join(f'out_type{i}' if not type_has_unknown_cats(out_types[i]) else f'arg{i}' for i in range(n_out_vars))}{comma})"
        # pass number of table arrays since it's necessary for logical column index
        # calculation but input table may be optimized out
        func_text += f"  out_data = cpp_table_to_py_data(out_cpp_table, out_col_inds, {out_types_str}, total_rows_np[0], {sort_node.num_table_arrays})\n"
    else:
        py_to_cpp_inds = {k: i for i, k in enumerate(out_cpp_col_inds)}

        arr_vars = []
        for i in range(n_cols):
            out_ind = py_to_cpp_inds.get(i, -1)
            if out_ind != -1:
                # unknown categorical arrays need actual value to convert array from C++
                out_type = (
                    f"out_type{i}"
                    if not type_has_unknown_cats(out_types[i])
                    else f"arg{i}"
                )
                func_text += f"  out{i} = array_from_cpp_table(out_cpp_table, {out_ind}, {out_type})\n"
                arr_vars.append(f"out{i}")

        comma = "," if len(arr_vars) == 1 else ""
        out_rets_tup = f"({', '.join(arr_vars)}{comma})"
        func_text += f"  out_data = {out_rets_tup}\n"

    func_text += "  delete_table(out_cpp_table)\n"
    func_text += "  return out_data\n"

    return func_text, {
        "in_col_inds": MetaType(tuple(in_cpp_col_inds)),
        "out_col_inds": MetaType(tuple(out_cpp_col_inds)),
    }


def _get_cpp_col_ind_mappings(key_inds, dead_var_inds, dead_key_var_inds, n_cols):
    """get mapping of indices in logical input columns to C++ (keys in order, then live
    input data columns), and mapping of indices from C++ to logical output columns (live
    keys are returned in order, then live output data).
    NOTE: dead logical column indices are not removed from logical indices

    Example input with 6 logical columns:
    key_inds = [3, 1, 4]
    dead_key_var_inds = [1]
    dead_var_inds = [2]
    in_cpp_col_inds = [3, 1, 4, 0, 5]
    out_cpp_col_inds: [3, 4, 0, 5]
    dead_keys = [1]

    Args:
        key_inds (list(int)): list of key indices from input in order (logical indices,
            including dead columns)
        dead_var_inds (set(int)): indices of dead input columns (logical)
        dead_key_var_inds (set(int)): indices of dead input keys (logical indices,
            including other dead columns)
        n_cols (int): number of logical input columns in sort node

    Returns:
        tuple(list(int), array(int), list(int)):
            logical indices of input columns to pass to C++,
            mapping of C++ output to logical output columns,
            list of dead C++ keys
    """

    # indices of logical columns that are passed to C++
    # keys go first in order as expected in sort_values_table
    in_cpp_col_inds = []
    out_cpp_col_inds = []
    dead_keys = []
    for k, i in enumerate(key_inds):
        in_cpp_col_inds.append(i)
        if i in dead_key_var_inds:
            dead_keys.append(k)
        else:
            out_cpp_col_inds.append(i)

    # live data columns
    for i in range(n_cols):
        if i in dead_var_inds or i in key_inds:
            continue

        in_cpp_col_inds.append(i)
        out_cpp_col_inds.append(i)

    return in_cpp_col_inds, out_cpp_col_inds, dead_keys


def sort_table_column_use(
    sort_node, block_use_map, equiv_vars, typemap, table_col_use_map
):
    """Compute column uses in input table of sort based on output table's uses. The
    input uses are the same as output, except that key columns are always used.

    Args:
        sort_node (Sort): Sort node to process
        block_use_map (Dict[str, Tuple[Set[int], bool, bool]]): column uses for current
            block.
        equiv_vars (Dict[str, Set[str]]): Dictionary
            mapping table variable names to a set of
            other table name aliases.
        typemap (dict[str, types.Type]): typemap of variables
        table_col_use_map (Dict[int, Dict[str, Tuple[Set[int], bool, bool]]]):
            Dictionary mapping block numbers to a dictionary of table names
            and "column uses". A column use is represented by the triple
                - used_cols: Set of used column numbers
                - use_all: Flag for if all columns are used. If True used_cols
                    is garbage
                - cannot_del_columns: Flag indicate this table is used in
                    an unsupported operation (e.g. passed to a DataFrame)
                    and therefore no columns can be deleted.
    """
    if (
        not sort_node.is_table_format
        or sort_node.in_vars[0] is None
        or sort_node.out_vars[0] is None
    ):
        return

    rhs_table = sort_node.in_vars[0].name
    rhs_key = (rhs_table, None)
    lhs_table = sort_node.out_vars[0].name
    lhs_key = (lhs_table, None)

    (
        orig_used_cols,
        orig_use_all,
        orig_cannot_del_cols,
    ) = block_use_map[rhs_key]

    # skip if input already uses all columns or cannot delete the table
    if orig_use_all or orig_cannot_del_cols:
        return

    # get output's uses
    (
        used_cols,
        use_all,
        cannot_del_cols,
    ) = _compute_table_column_uses(lhs_key, table_col_use_map, equiv_vars)

    # key columns are always used in sorting
    table_key_set = {i for i in sort_node.key_inds if i < sort_node.num_table_arrays}

    block_use_map[rhs_key] = (
        orig_used_cols | used_cols | table_key_set,
        use_all or cannot_del_cols,
        False,
    )


ir_extension_table_column_use[Sort] = sort_table_column_use


def sort_remove_dead_column(sort_node, column_live_map, equiv_vars, typemap):
    """Remove dead table columns from Sort node (if in table format). Updates
    dead_key_var_inds and dead_var_inds sets based on used column info from dead table
    column pass.

    Args:
        sort_node (Sort): Sort node to update
        column_live_map (Dict[str, Tuple[Set[int], bool, bool]]): column uses of each
            table variable for current block.
        equiv_vars (Dict[str, Set[str]]): Dictionary
            mapping table variable names to a set of
            other table name aliases.
        typemap (dict[str, types.Type]): typemap of variables
    """
    if not sort_node.is_table_format or sort_node.out_vars[0] is None:
        return False

    n_table_cols = sort_node.num_table_arrays
    lhs_table = sort_node.out_vars[0].name
    lhs_table_key = (lhs_table, None)

    used_columns = _find_used_columns(
        lhs_table_key, n_table_cols, column_live_map, equiv_vars
    )

    # None means all columns are used so we can't prune any columns
    if used_columns is None:
        return False

    dead_columns = set(range(n_table_cols)) - used_columns

    table_key_set = {i for i in sort_node.key_inds if i < n_table_cols}
    new_dead_keys = sort_node.dead_key_var_inds | (dead_columns & table_key_set)
    new_dead_vars = sort_node.dead_var_inds | (dead_columns - table_key_set)
    removed = (new_dead_keys != sort_node.dead_key_var_inds) | (
        new_dead_vars != sort_node.dead_var_inds
    )

    sort_node.dead_key_var_inds = new_dead_keys
    sort_node.dead_var_inds = new_dead_vars

    return removed


remove_dead_column_extensions[Sort] = sort_remove_dead_column
