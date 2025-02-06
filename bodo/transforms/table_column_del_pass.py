"""
Updates the function IR to include decref on individual columns
when they are no longer used. This enables garbage collecting
single columns when tables are represented by a single variable.
"""

import copy
import typing as pt
from collections import defaultdict

import numba
from numba.core import ir, types
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.ir_utils import build_definitions, find_topo_order, guard

import bodo
from bodo.hiframes.table import TableType, del_column
from bodo.utils.del_column_utils import (
    get_table_used_columns,
    is_table_use_column_ops,
)
from bodo.utils.transform import compile_func_single_block
from bodo.utils.typing import get_overload_const_int, is_overload_constant_int
from bodo.utils.utils import is_assign, is_call, is_expr


class TableColumnDelPass:
    """
    This pass determine where in a program's execution TableTypes
    can remove column that must be loaded, but aren't needed for
    the full execution of the program. The goal of this pass is to
    be able reduce Bodo's memory footprint when a column isn't needed
    for the whole program. For example:

    T = Table(arr0, arr1)
    print(T[0])
    # T[0] isn't used again
    ...
    # Much later in the program
    return T[1]

    Here T is live for nearly the whole program, but T[0] is
    only needed at the beginning. This possibly wastes memory
    if T[0] only gets decrefed when T is deallocated, so
    this pass aims to eliminate this column earlier in a program's
    execution. To do this, we apply a 4 part algorithm.

    1. Column Liveness: Determine what columns may be live in each block

    2. Alias Grouping: Group variables that refer to the same table together.

    3. Determine Column Changes: Determine where in the control flow columns
        are no longer needed from 1 block to the next.

    4. Remove Columns: Insert code into the IR to remove the columns.

    For a more complete discussion of the steps of this algorithm, please
    refer to its confluence page:
    https://bodo.atlassian.net/wiki/spaces/B/pages/920354894/Table+Column+Decref+Pass

    Del placement is inspired by Numba's compute_dead_maps
    https://github.com/numba/numba/blob/5fc9d3c56da4e4c6aef7189e588ce9c44263d4a6/numba/core/analysis.py#L118,
    in particular 'internal_dead_map' and 'escaping_dead_map'.

    Actual del insertion is inspired by Numba's PostProcessor._patch_var_dels
    https://github.com/numba/numba/blob/5fc9d3c56da4e4c6aef7189e588ce9c44263d4a6/numba/core/postproc.py#L178
    """

    def __init__(self, func_ir, typingctx, targetctx, typemap, calltypes):
        """
        Initialization information. Some fields are included because
        they are needed to call 'compile_func_single_block' and are
        otherwise unused.
        """
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.typemap = typemap
        self.calltypes = calltypes
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc

    def run(self):
        # Collect information that are needed for various stages
        # of the algorithm
        f_ir = self.func_ir
        typemap = self.typemap
        cfg = compute_cfg_from_blocks(f_ir.blocks)
        # Get the livemap for variables.
        livemap = numba.core.postproc.VariableLifetime(f_ir.blocks).livemap

        # Step 1: Column Liveness. Determine which columns may be live
        column_live_map, equiv_vars = compute_column_liveness(
            cfg, f_ir.blocks, f_ir, typemap
        )

        # TableColumnDelPass operates under the assumption that aliases are transitive,
        # but this assumption has since been changed. For right now, we simply convert
        # this aliases to a transitive representation.
        # See https://bodo.atlassian.net/jira/software/projects/BE/boards/4/backlog?selectedIssue=BE-3028

        # copy to avoid changing size during iteration
        old_alias_map = copy.deepcopy(equiv_vars)
        # combine all aliases transitively
        for v in old_alias_map:
            equiv_vars[v].add(v)
            for w in old_alias_map[v]:
                equiv_vars[v] |= equiv_vars[w]
            for w in old_alias_map[v]:
                equiv_vars[w] = equiv_vars[v]

        # Step 2: Alias Grouping. Determine which variables are the same table
        # and ensure they share live columns.
        alias_sets, alias_set_liveness_map = self.compute_alias_grouping(
            cfg,
            f_ir.blocks,
            column_live_map,
            equiv_vars,
        )
        # Step 3: Determine Column Changes. Find the places in the control flow
        # where columns can be deleted. Depending on liveness, columns will either
        # be deleted starting from the end of a block (internal_dead) or at the
        # front of a block (escaping_dead).
        internal_dead, escaping_dead = self.compute_column_del_locations(
            cfg,
            f_ir.blocks,
            alias_sets,
            alias_set_liveness_map,
            typemap,
        )
        updated = False
        # Step 4: Remove columns. Compute the actual columns decrefs and
        # update the IR.
        updated = self.insert_column_dels(
            cfg,
            f_ir.blocks,
            f_ir,
            typemap,
            internal_dead,
            escaping_dead,
            alias_sets,
            equiv_vars,
            livemap,
        )
        return updated

    def compute_alias_grouping(self, cfg, blocks, column_live_map, equiv_vars):
        """Compute the actual columns used in each block so
        we can generate del_column calls. This function groups
        common tables together in an alias set that can be used to
        keep information active.

        It returns 2 values:

            'alias_sets':    A dictionary of unique alias sets (table_key -> aliases)

            'liveness_map':  A dictionary mapping each block to a
                             dictionary {table_key -> set(live_columns)}
                             This table name is the same key for alias_sets.

        The main reason we opt to reduce the equiv_vars to unique alias sets
        is to track which columns have already had del_column calls introduced.
        Consider the following example

        Block 0:
            T = table(arr0, arr1)
            T1 = T
            T2 = T
            jump 1
        Block 1:
            arr1 = get_table_data(T1, 0)
            arr2 = get_table_data(T2, 0)

        Here we need to insert del_column exactly once per set. If we try and input del_column
        after 'get_table_data(T1, 0)' we will get an incorrect result (as the array will be already deleted).

        For more information see:
        https://bodo.atlassian.net/wiki/spaces/B/pages/920354894/Table+Column+Decref+Pass#Alias-Grouping
        """
        # Compute the set of distinct alias groups.
        # The source node will contain which variables use
        # cannot_del_cols, which we must omit.
        source_live_cols = column_live_map[cfg.entry_point()]
        alias_sets = _find_table_alias_sets(source_live_cols, equiv_vars)
        # {block_offset -> {table_name: set(used_columns)}}
        # The alias_sets and table_name in livecol_map share a common representative
        # variable.
        livecol_map = {}
        for offset in blocks.keys():
            block_column_liveness = column_live_map[offset]
            # Group live columns within an alias set.
            livecol_map[offset] = set_dict = {}
            for table_key in alias_sets.keys():
                used_columns, use_all, _ = get_live_column_nums_block(
                    block_column_liveness, equiv_vars, table_key
                )
                # Convert back to a set for easier comparison between blocks.
                set_dict[table_key] = (used_columns, use_all)

        return alias_sets, livecol_map

    def compute_column_del_locations(
        self, cfg, blocks, alias_sets, alias_set_livecol_map, typemap
    ):
        """
        Compute where to insert the decrefs. The algorithm for the steps here:
        https://bodo.atlassian.net/wiki/spaces/B/pages/920354894/Table+Column+Decref+Pass#Determine-Column-Changes

        There are two maps that are returned 'internal_dead' and 'escaping_dead'

        If a block's set contains a column not found in any successors,
        we want to delete it at the end of that block ('internal_dead').

        If a block's set contains a column found in at least 1 successor, but not all,
        we want to delete the column at the start of the successors that do not
        use the column ('escaping_dead').

        These notable inputs have following format:
            alias_sets: {table_name -> set(table_aliases)}.
                Each set is unique with 1 representative
                table that is not found in the set.

            alias_set_livecol_map: {block_offset -> {table_name -> set(use_columns)}}
                The table names are consistent with alias_sets.

                use_columns consists all columns that are live at the start of the block
                or is defined within the block for the set of aliases represented by table_name.
        """
        internal_dead = {}  # {block -> {set_table -> set of del columns}}
        escaping_dead = defaultdict(
            lambda: defaultdict(set)
        )  # {block -> {set_table -> set of del columns}}
        for offset in blocks.keys():
            internal_dead[offset] = col_back_map = {}
            block_livecols = alias_set_livecol_map[offset]
            for table_key, aliases in alias_sets.items():
                # Get the type information for the current group.
                var_type = typemap[table_key[0]]
                if table_key[1] is None:
                    # The variable is a table
                    table_type = var_type
                else:
                    # The variable is a tuple containing the table.
                    table_type = var_type.types[table_key[1]]
                # Determine the variables to delete in this block
                curr_cols, use_all = block_livecols[table_key]
                # Find all columns that are live in any successor
                combined_succ_livecols = set()
                # Does any successor use all
                any_succ_use_all = False
                # Do all successors use all
                all_succ_use_all = True
                for label, _ in cfg.successors(offset):
                    succ_livecols, succ_use_all = alias_set_livecol_map[label][
                        table_key
                    ]
                    any_succ_use_all = any_succ_use_all or succ_use_all
                    all_succ_use_all = all_succ_use_all and succ_use_all

                    if not any_succ_use_all:
                        combined_succ_livecols = combined_succ_livecols | succ_livecols

                # Delete columns that aren't live in any successor.
                if not any_succ_use_all:
                    if use_all:
                        # If we switch from use_all to use some
                        # we need to generate the set.
                        curr_cols = set(range(len(table_type.arr_types)))
                    # Delete any columns not found in this block.
                    del_curr = curr_cols - combined_succ_livecols
                else:
                    del_curr = set()
                col_back_map[table_key] = del_curr

                # Now determine which column can be deleted for each successor.
                # Some blocks may still use all columns while others may not.
                if not all_succ_use_all:
                    if any_succ_use_all:
                        # No columns were killed internally
                        escaping_live_set = set(range(len(table_type.arr_types)))
                    else:
                        escaping_live_set = curr_cols - del_curr
                    for label, _ in cfg.successors(offset):
                        # For each successor delete columns that only need
                        # to be deleted from that column
                        succ_livecols, succ_use_all = alias_set_livecol_map[label][
                            table_key
                        ]
                        if not succ_use_all:
                            escaping_dead[label][table_key] |= (
                                escaping_live_set - succ_livecols
                            )
        return internal_dead, escaping_dead

    def insert_column_dels(
        self,
        cfg,
        blocks,
        func_ir,
        typemap,
        internal_dead,
        escaping_dead,
        alias_sets,
        equiv_vars,
        livemap,
    ):
        """
        Insert the decrefs + set the column to NULL.
        If we have a internal dead column, we traverse backwards until
        we find each relevant table use (get_dataframe_data and table ops)
        and insert the del immediately afterwards.

        If we have a escaping dead column we just insert at the front.

        In each situation we need to find a live variable to decref the column. If no
        variable exists then we don't delete the column because the whole table will
        be deleted anyways.

        https://bodo.atlassian.net/wiki/spaces/B/pages/920354894/Table+Column+Decref+Pass#Remove-Columns
        """

        updated = False
        for offset, block in blocks.items():
            insert_front = []
            escaping_cols = escaping_dead[offset]
            if escaping_cols:
                block_livemap = livemap[offset]
                # Generate the code to decref all columns.
                args = ", ".join([f"arg{i}" for i in range(len(escaping_cols.keys()))])
                ctr = 0
                var_names = []
                func_lines = []
                # Track globals to pass for deleting columns
                col_globals: dict[str, bodo.utils.typing.MetaType] = {}
                for table_key, columns in escaping_cols.items():
                    used_var_key = get_livevar_key(
                        table_key, alias_sets[table_key], block_livemap
                    )
                    if used_var_key and columns:
                        # We only add this table if there is at least 1 live
                        # var and columns is not empty. See test_table_dead_var
                        updated = True
                        var_names.append(used_var_key[0])
                        col_globals[f"cols_to_delete_{ctr}"] = (
                            bodo.utils.typing.MetaType(tuple(sorted(columns)))
                        )
                        func_lines.append(
                            f"    del_column(arg{ctr}, cols_to_delete_{ctr})\n"
                        )
                        ctr += 1
                if var_names:
                    args = ", ".join([f"arg{i}" for i in range(len(var_names))])
                    func_text = f"def del_columns({args}):\n"
                    func_text += "".join(func_lines)
                    # Only compile the function if at least 1 table needs to be deleted.
                    new_stmts = self._compile_del_column_function(
                        func_text, var_names, col_globals
                    )
                    # Insert into the front of the block.
                    insert_front.extend(new_stmts)

            internal_cols = internal_dead[offset]
            new_body = block.body
            if internal_cols:
                # Find where to decref each column.
                new_body = []
                for stmt in reversed(block.body):
                    # Search for get_table_data calls to remove tables.
                    if (
                        isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == "call"
                    ):
                        rhs = stmt.value
                        fdef = guard(numba.core.ir_utils.find_callname, func_ir, rhs)
                        # Only eliminate columns once we find the get_table_data call
                        if is_table_use_column_ops(fdef, rhs.args, typemap):
                            # Note: This won't handle the tuple case. However we don't
                            # support any operation that goes Table -> Tuple, so this
                            # is always safe. If/when we support build_tuple we will
                            # need to handle this case.
                            table_var_name = rhs.args[0].name
                            table_var_key = (table_var_name, None)
                            col_nums = get_table_used_columns(fdef, rhs, typemap)
                            # Determine the table key.
                            if table_var_key in internal_cols:
                                cols = internal_cols[table_var_key]
                            else:
                                # If the name we found isn't the representative
                                # for the alias, find the representative if it exists.
                                cols = set()
                                s = equiv_vars[table_var_key]
                                for table_name in sorted(s):
                                    if table_name in internal_cols:
                                        cols = internal_cols[table_name]
                                        break
                            if col_nums is None:
                                # If we cannot identify which columns to prune,
                                # we must prune all at this step as otherwise
                                # an earlier operation could incorrectly attempt
                                # to remove the column.
                                deleted_cols = cols
                            else:
                                deleted_cols = col_nums & cols
                            if deleted_cols:
                                updated = True
                                func_text = "def del_columns(table_arg):\n"
                                # Track globals to pass for deleting columns. This
                                # is one element but we pass a dict for consistency with
                                # the block start case.
                                col_globals: dict[str, bodo.utils.typing.MetaType] = {
                                    "cols_to_delete": bodo.utils.typing.MetaType(
                                        tuple(sorted(deleted_cols))
                                    )
                                }
                                func_text += (
                                    "    del_column(table_arg, cols_to_delete)\n"
                                )
                                # Compile the function
                                new_stmts = self._compile_del_column_function(
                                    func_text, [table_var_name], col_globals
                                )
                                # Insert stmts in reverse order because we reverse the block
                                new_body.extend(list(reversed(new_stmts)))
                                # Mark the column as removed. Note each alias set contains
                                # a unique set per block that is not copied anywhere. We
                                # update this set so we don't try remove a column earlier
                                # in the block.
                                cols.difference_update(deleted_cols)

                    new_body.append(stmt)
                # We insert into the body in reverse order, so reverse it again.
                new_body = list(reversed(new_body))
            block.body = insert_front + new_body
        return updated

    def _compile_del_column_function(self, func_text, var_names, col_globals):
        """
        Helper function to compile and return the statements
        from compile each decref function.
        """
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        # Create an ir.Var for each var_name
        arg_vars = [ir.Var(None, var_name, self.curr_loc) for var_name in var_names]
        impl = loc_vars["del_columns"]
        return compile_func_single_block(
            impl,
            tuple(arg_vars),
            None,
            typing_info=self,
            extra_globals={"del_column": del_column, **col_globals},
        )


def _find_table_alias_sets(source_live_cols, equiv_vars):
    """
    Given a live map for the source block and the equiv_vars dictionary,
    returns a dictionary of all unique table alias sets. We remove any
    sets in which a table data structure is shared with a DataFrame
    (as indicated by the cannot_del_cols flag). This ensures that those
    tables will never have columns deleted.

    This is essential because we cannot remove columns from a table if
    there is a DataFrame in the IR that reuses the table (as a DataFrame
    does not support deleted columns). We do not keep tracks of the DataFrame
    vars in the alias_sets, only tables.
    """
    seen_keys = set()
    alias_sets = {}
    for table_key in source_live_cols.keys():
        if table_key not in seen_keys:
            # Mark all members of the alias set as seen
            seen_keys.add(table_key)
            seen_keys.update(equiv_vars[table_key])
            _, _, cannot_del_cols = get_live_column_nums_block(
                source_live_cols, equiv_vars, table_key
            )
            if not cannot_del_cols:
                # Only add the set if it can delete columns.
                alias_sets[table_key] = equiv_vars[table_key]
    return alias_sets


# Global dictionaries for IR extensions to handle dead column analysis
remove_dead_column_extensions = {}
ir_extension_table_column_use = {}


def get_livevar_key(table_key, aliases, block_livemap):
    """
    Given a set of alias keys, return the key
    of a variable live in the block_livemap or None
    if no variable is found.

    Note we require the index is None for safe column deletion because
    we do not support tuples yet.
    """
    # Sort to ensure this is with multiple ranks.
    if table_key[0] in block_livemap and table_key[1] is None:
        return table_key
    for alias in sorted(aliases):
        if alias[0] in block_livemap and alias[1] is None:
            return alias
    return None


def compute_column_liveness(cfg, blocks, func_ir, typemap):
    """
    Compute which columns may be live in each block. These are computed
    according to the DataFlow algorithm found in this confluence document

    https://bodo.atlassian.net/wiki/spaces/B/pages/912949249/Dead+Column+Elimination+with+Large+Numbers+of+Columns

    We keep the use and live computations as closures to make these functions "private"
    to this function, which should always be called instead.
    """
    column_usemap, equiv_vars = _compute_table_column_use(blocks, func_ir, typemap)
    column_live_map = _compute_table_column_live_map(cfg, column_usemap)
    return column_live_map, equiv_vars


def _compute_table_column_use(blocks, func_ir, typemap):
    """
    Find table column use per block. This returns two dictionaries,
    one that tracks columns used for each block, 'table_col_use_map'
    and a dictionary that tracks aliases 'equiv_vars'.

    'table_col_use_map' values are a tuple of 3 values:

        1. A set of columns directly used.
        2. A boolean flag for if all column are used. This is an optimization
        to reduce memory any operation requires all columns,
        3. A boolean flag indicating if an operation is unknown and therefore
        it is not safe to prune any columns. This is used when tables become
        part of DataFrames.

    'equiv_vars' maps each table name to all other tables that function as an alias.
    """
    # For maps below key_type is (var_name, Optional(int64)) where the int64 is the index if we have
    # a tuple and None if we have a table.

    # keep track of potential aliases for tables.
    # key_type -> Set(key_type)
    equiv_vars = defaultdict(set)
    table_col_use_map = {}  # { block offset -> dict(key_type -> tuple(set(used_column_numbers), use_all, cannot_del_cols)}
    # See table filter note below regarding reverse order
    for offset in reversed(find_topo_order(blocks)):
        ir_block = blocks[offset]
        table_col_use_map[offset] = block_use_map = defaultdict(
            lambda: (set(), False, False)
        )
        for stmt in reversed(ir_block.body):
            # IR extensions that impact column uses.
            if type(stmt) in ir_extension_table_column_use:
                f = ir_extension_table_column_use[type(stmt)]
                # We assume that f checks types if necessary
                # and performs any required actions.
                f(stmt, block_use_map, equiv_vars, typemap, table_col_use_map)
                continue

            # If we have an assign we don't want to mark that variable as used.
            lhs_name = None

            # Check for assigns. This should include all basic usages
            # (get_dataframe_data + simple assign)
            if isinstance(stmt, ir.Assign):
                rhs = stmt.value
                lhs_name = stmt.target.name
                # Key for Operations that return a table
                table_lhs_key = (lhs_name, None)
                # Type for the lhs
                lhs_type = typemap[lhs_name]
                # Handle simple assignments (i.e. df2 = df1)
                # This should add update the equiv_vars
                # Arrow reader is equivalent to tables for column elimination purposes
                if isinstance(rhs, ir.Var) and isinstance(
                    lhs_type,
                    (TableType, bodo.io.arrow_reader.ArrowReaderType),
                ):
                    lhs_key = (lhs_name, None)
                    rhs_key = (rhs.name, None)
                    _update_equiv_set(equiv_vars, lhs_key, rhs_key)
                    continue

                # Handle simple assignments for tuples
                if isinstance(rhs, ir.Var) and isinstance(lhs_type, types.BaseTuple):
                    # Handle tuple assignments. For every element in the tuple
                    # that is a TableType we equate the lhs and rhs.
                    for i, elem in enumerate(lhs_type.types):
                        if isinstance(elem, TableType):
                            lhs_key = (lhs_name, i)
                            rhs_key = (rhs.name, i)
                            _update_equiv_set(equiv_vars, lhs_key, rhs_key)
                    continue

                # Exhaust iter is basically an assign for tuples
                if is_expr(rhs, "exhaust_iter") and isinstance(
                    lhs_type, types.BaseTuple
                ):
                    # Fetch the number of elements from the exhaust_iter
                    num_elems = rhs.count
                    elem_types = lhs_type.types
                    assert num_elems <= len(elem_types), (
                        "Internal Error: Invalid exhaust_iter count"
                    )
                    for i in range(num_elems):
                        if isinstance(elem_types[i], TableType):
                            lhs_key = (lhs_name, i)
                            rhs_key = (rhs.value.name, i)
                            _update_equiv_set(equiv_vars, lhs_key, rhs_key)
                    continue

                # Getitem for tuples should equate two tables. If the index is not
                # static (which shouldn't occur, but is technically possible for UniTuple,
                # we mark all tables in the tuple as possibly identical).
                if (
                    is_expr(stmt.value, "getitem")
                    or is_expr(stmt.value, "static_getitem")
                ) and isinstance(typemap[rhs.value.name], types.BaseTuple):
                    # Mark as a known operation regardless of if we extract the table.
                    if isinstance(lhs_type, TableType):
                        lhs_key = (lhs_name, None)
                        index_var = stmt.value.index
                        if stmt.value.op == "static_getitem":
                            index = index_var
                        else:
                            index_type = typemap[index_var.name]
                            if is_overload_constant_int(index_type):
                                index = get_overload_const_int(index_type)
                            else:  # pragma: no cover
                                # We can't find the index type.
                                index = None
                        if index is None:  # pragma: no cover
                            # Mark every table in the tuple as equivalent
                            # because we can't determine which element
                            # we are accessing. Again we don't expect this path
                            # to be reached.
                            tuple_elem_types = typemap[rhs.value.name]
                            for i in range(len(tuple_elem_types)):
                                if isinstance(tuple_elem_types[i], TableType):
                                    lhs_key = (lhs_name, None)
                                    rhs_key = (rhs.value.name, i)
                                    _update_equiv_set(equiv_vars, lhs_key, rhs_key)
                        else:
                            rhs_key = (rhs.value.name, index)
                            _update_equiv_set(equiv_vars, lhs_key, rhs_key)
                    continue

                # Handle calls to get_table_data
                if is_call(stmt.value):
                    rhs = stmt.value
                    fdef = guard(numba.core.ir_utils.find_callname, func_ir, rhs)
                    if fdef == ("get_table_data", "bodo.hiframes.table"):
                        table_var_name = rhs.args[0].name
                        assert isinstance(typemap[table_var_name], TableType), (
                            "Internal Error: Invalid get_table_data call"
                        )
                        col_num = typemap[rhs.args[1].name].literal_value
                        table_key = (table_var_name, None)
                        col_num_set = block_use_map[table_key][0]
                        col_num_set.add(col_num)
                        continue
                    elif fdef == (
                        "get_dataframe_table",
                        "bodo.hiframes.pd_dataframe_ext",
                    ):
                        # If there is a get_dataframe_table in the IR then the source of
                        # this table is a live DataFrame. As a result, it is not safe to
                        # delete columns from this table or it aliases (although copies
                        # that create a new list can have their columns deleted). This is
                        # equivalent to the behavior with tuple format when we cannot
                        # optimize out the DataFrame.
                        block_use_map[table_lhs_key] = (set(), True, True)
                        continue
                    elif fdef in (
                        ("set_table_data", "bodo.hiframes.table"),
                        (
                            "set_table_data_null",
                            "bodo.hiframes.table",
                        ),
                    ):
                        # Set table data uses pass to the input table except for
                        # the set column.
                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        # If the RHS uses all columns already we can skip.
                        (
                            orig_used_cols,
                            orig_use_all,
                            orig_cannot_del_cols,
                        ) = block_use_map[rhs_key]
                        # If the RHS uses all columns or cannot delete the table we
                        # can skip.
                        if orig_use_all or orig_cannot_del_cols:
                            continue

                        (
                            used_cols,
                            use_all,
                            cannot_del_cols,
                        ) = _compute_table_column_uses(
                            table_lhs_key, table_col_use_map, equiv_vars
                        )
                        # If the column being set is used in the original table it does
                        # not refer to the same column. As a result we remove it from
                        # the used_cols.
                        set_col_num = get_overload_const_int(typemap[rhs.args[1].name])
                        used_cols.discard(set_col_num)
                        block_use_map[rhs_key] = (
                            orig_used_cols | used_cols,
                            use_all or cannot_del_cols,
                            False,
                        )
                        continue
                    elif fdef == ("len", "builtins"):
                        # Skip ops that shouldn't impact the number of columns. Len
                        # requires some column in the table, but no particular column.
                        continue
                    elif fdef == (
                        "generate_mappable_table_func",
                        "bodo.utils.table_utils",
                    ):
                        # Handle mappable operations table operations. These act like
                        # an alias.
                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        used_cols, use_all, cannot_del_cols = _generate_rhs_use_map(
                            rhs_key,
                            block_use_map,
                            table_col_use_map,
                            equiv_vars,
                            table_lhs_key,
                        )
                        block_use_map[rhs_key] = (used_cols, use_all, cannot_del_cols)
                        continue
                    elif fdef == (
                        "table_astype",
                        "bodo.utils.table_utils",
                    ):
                        # While astype may or may not make a copy, the
                        # actual astype operation never modifies the contents
                        # of the columns. This operation matches the input and
                        # output tables, but it does not use any additional columns.
                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        used_cols, use_all, cannot_del_cols = _generate_rhs_use_map(
                            rhs_key,
                            block_use_map,
                            table_col_use_map,
                            equiv_vars,
                            table_lhs_key,
                        )
                        block_use_map[rhs_key] = (used_cols, use_all, cannot_del_cols)
                        continue
                    # handle table filter like T2 = T1[ind]
                    elif (
                        isinstance(fdef, tuple)  # fdef can be None
                        and fdef[1] == "bodo.hiframes.table"
                        and fdef[0] in ("table_filter", "table_local_filter")
                    ):
                        # NOTE: column uses of input T1 are the same as output T2.
                        # Here we simply traverse the IR in reversed order and update uses,
                        # which works because table filter variables are internally
                        # generated variables and have a single definition without control
                        # flow. Otherwise, we'd need to update uses iteratively.

                        # NOTE: We must search the entire table_col_use_map at this point
                        # because we haven't updated column usage/use_all from successor
                        # blocks yet.

                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        used_cols, use_all, cannot_del_cols = _generate_rhs_use_map(
                            rhs_key,
                            block_use_map,
                            table_col_use_map,
                            equiv_vars,
                            table_lhs_key,
                        )
                        block_use_map[rhs_key] = (used_cols, use_all, cannot_del_cols)
                        continue

                    # handle selecting a subset of columns like T2 = T1[col_nums].
                    # This is the table equivalent of df2 = df[["A", "B", "C", ...]]
                    elif fdef == ("table_subset", "bodo.hiframes.table"):
                        # Taking the a table subset changes the meaning of the array
                        # numbers. As a result, we will need to remap columns used
                        # in the input to the output. For example if we code that
                        # looks like
                        #
                        # T1 = (arr0, arr1, arr2, arr3)
                        # T2 = table_subset(T1, [2, 2, 0])
                        # get_dataframe_data(T2, 0)
                        # get_dataframe_data(T2, 1)
                        #
                        # Then T2 would see that is uses columns (0, 1).
                        # To determine T1's uses remap back to the original
                        # index, computing [2, 2] or just column 2.

                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        # If the RHS uses all columns already we can skip.
                        orig_used_cols, use_all, cannot_del_cols = block_use_map[
                            rhs_key
                        ]
                        # If the RHS uses all columns or cannot delete the table we
                        # can skip.
                        if use_all or cannot_del_cols:
                            continue
                        # Create the mapping between the input the types and the output
                        # types.
                        (
                            used_cols,
                            use_all,
                            cannot_del_cols,
                        ) = _compute_table_column_uses(
                            table_lhs_key, table_col_use_map, equiv_vars
                        )
                        # The used columns are stored in arg1, which is
                        # a typeref MetaType
                        idx_list = typemap[rhs.args[1].name].instance_type.meta
                        if use_all or cannot_del_cols:
                            # If we use all or cannot delete columns for the output
                            # we add every input column. We can still delete columns
                            # from the input potentially because this creates new
                            # lists.
                            final_used_cols = set(idx_list)
                        else:
                            final_used_cols = set()
                            for i, idx in enumerate(idx_list):
                                if i in used_cols:
                                    final_used_cols.add(idx)
                        block_use_map[rhs_key] = (
                            orig_used_cols | final_used_cols,
                            False,
                            False,
                        )
                        continue

                    # Table_concat is used by melt
                    elif fdef == ("table_concat", "bodo.utils.table_utils"):
                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        used_cols, use_all, cannot_del_cols = block_use_map[rhs_key]
                        if not (use_all or cannot_del_cols):
                            # The used columns cannot change for this operation
                            # and are found in the type for argument 1.
                            used_cols_type = typemap[rhs.args[1].name]
                            used_cols_type = used_cols_type.instance_type
                            block_use_map[rhs_key] = (
                                used_cols | set(used_cols_type.meta),
                                use_all,
                                cannot_del_cols,
                            )
                        continue
                    elif fdef == ("generate_table_nbytes", "bodo.utils.table_utils"):
                        # generate_table_nbytes uses all columns. These columns
                        # can still be deleted however.
                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        used_cols, use_all, cannot_del_cols = block_use_map[rhs_key]
                        block_use_map[rhs_key] = set(), True, cannot_del_cols
                        continue
                    elif fdef == ("py_data_to_cpp_table", "bodo.libs.array"):
                        # after lowering the Sort IR node into py_data_to_cpp_table and
                        # other calls, the uses of columns cannot change anymore.
                        # But the unused columns can be deleted.
                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        if isinstance(typemap[rhs_table], TableType):
                            used_cols, use_all, cannot_del_cols = block_use_map[rhs_key]
                            if not (use_all or cannot_del_cols):
                                in_cols = typemap[rhs.args[2].name].instance_type.meta
                                # trim logical column uses that are not in the table (are
                                # in extra arrays argument)
                                n_table_cols = len(typemap[rhs_table].arr_types)
                                in_cols_set = {i for i in in_cols if i < n_table_cols}
                                block_use_map[rhs_key] = (
                                    used_cols | in_cols_set,
                                    use_all,
                                    cannot_del_cols,
                                )
                        continue

                    # first argument can be a table or tuple of arrays or None (for dead
                    # table input)
                    elif fdef == (
                        "logical_table_to_table",
                        "bodo.hiframes.table",
                    ) and isinstance(typemap[rhs.args[0].name], TableType):
                        rhs_table = rhs.args[0].name
                        rhs_key = (rhs_table, None)
                        orig_used_cols, use_all, cannot_del_cols = block_use_map[
                            rhs_key
                        ]
                        # Skip if input table uses all columns or cannot be deleted
                        if use_all or cannot_del_cols:
                            continue

                        # Create the mapping between the input the types and the output
                        # types.
                        (
                            lhs_used_cols,
                            lhs_use_all,
                            lhs_cannot_del_cols,
                        ) = _compute_table_column_uses(
                            table_lhs_key, table_col_use_map, equiv_vars
                        )

                        in_col_inds = typemap[rhs.args[2].name].instance_type.meta
                        n_in_table_arrs = len(typemap[rhs_table].arr_types)

                        if lhs_use_all or lhs_cannot_del_cols:
                            final_used_cols = {
                                i for i in in_col_inds if i < n_in_table_arrs
                            }
                        else:
                            final_used_cols = set()
                            for out_ind, in_ind in enumerate(in_col_inds):
                                if (
                                    out_ind in lhs_used_cols
                                    and in_ind < n_in_table_arrs
                                ):
                                    final_used_cols.add(in_ind)
                        block_use_map[rhs_key] = (
                            orig_used_cols | final_used_cols,
                            False,
                            False,
                        )
                        continue

                    elif fdef == ("read_arrow_next", "bodo.io.arrow_reader"):
                        # Set the iterator and out table as equivalent.
                        lhs_key = (lhs_name, 0)
                        rhs_key = (rhs.args[0].name, None)
                        _update_equiv_set(equiv_vars, lhs_key, rhs_key)
                        continue

                elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getattr":
                    if stmt.value.attr == "shape":
                        # Skip ops that shouldn't impact the number of columns. Shape
                        # can be computed from a combination of the type and the length
                        # of any column, but no particular column. This needs to be skipped
                        # because it is inserted by
                        continue

            for var in stmt.list_vars():
                # All unknown table uses marks columns as being not
                # safe to delete. This includes operations like inserting
                # a table into a DataFrame via init_dataframe(build_tuple(table,))
                # and operations like returns.
                #
                if var.name != lhs_name and isinstance(typemap[var.name], TableType):
                    table_key = (var.name, None)
                    # The table is used in an unexpected way so it is not safe to delete.
                    block_use_map[table_key] = (set(), True, True)
                elif var.name != lhs_name and isinstance(
                    typemap[var.name], types.BaseTuple
                ):
                    for i, typ in enumerate(typemap[var.name].types):
                        if isinstance(typ, TableType):
                            table_key = (var.name, i)
                            # The table is used in an unexpected way so it is not safe to delete.
                            block_use_map[table_key] = (set(), True, True)

    return table_col_use_map, equiv_vars


def _compute_table_column_live_map(cfg, column_uses):
    """
    Find columns that may be alive at the START of each block. Liveness here
    is approximate because we may have false positives (a column may be treated
    as live when it could be dead).

    For example if have

    Block B:
        T = Table(arr0, arr1)
        T[1]

    Then column 1 is considered live in Block B and all predecessors, even though T
    is defined in B. This is done to simplify the algorithm.

    We use a simple fix-point algorithm that iterates until the set of
    columns is unchanged for each block.

    This liveness structure is heavily influenced by compute_live_map inside Numba
    https://github.com/numba/numba/blob/944dceee2136ab55b595319aa19611e3699a32a5/numba/core/analysis.py#L60
    """

    def fix_point_progress(dct):
        """Helper function to determine if a fix-point has been reached.
        We detect this by determining the column numbers, use_all_flag,
        and cannot_del_cols values haven't changed.
        """
        results = []
        for vals in dct.values():
            results.append(tuple((len(v[0]), v[1], v[2]) for v in vals.values()))
        return tuple(results)

    def fix_point(fn, dct):
        """Helper function to run fix-point algorithm."""
        old_point = None
        new_point = fix_point_progress(dct)
        while old_point != new_point:
            fn(dct)
            old_point = new_point
            new_point = fix_point_progress(dct)

    def liveness(dct):
        """Find live columns.

        Push column usage backward.
        """
        for offset in dct:
            # Live columns here
            live_columns = dct[offset]
            for inc_blk, _ in cfg.predecessors(offset):
                for table_key, (uses, use_all, cannot_del_cols) in live_columns.items():
                    # Initialize the df if it doesn't exist or if liveness_tup[1]=True.
                    if table_key not in dct[inc_blk]:
                        dct[inc_blk][table_key] = (
                            uses.copy(),
                            use_all,
                            cannot_del_cols,
                        )
                    else:
                        pred_uses, pred_use_all, pred_cannot_del_cols = dct[inc_blk][
                            table_key
                        ]
                        if (
                            use_all
                            or cannot_del_cols
                            or pred_use_all
                            or pred_cannot_del_cols
                        ):
                            # If either use_all or cannot_del_cols
                            # we remove any column numbers to save memory.
                            dct[inc_blk][table_key] = (
                                set(),
                                use_all or pred_use_all,
                                cannot_del_cols or pred_cannot_del_cols,
                            )
                        else:
                            dct[inc_blk][table_key] = (uses | pred_uses, False, False)

    live_map = copy.deepcopy(column_uses)
    fix_point(liveness, live_map)
    return live_map


def remove_dead_columns(
    block,
    lives,
    equiv_vars,
    typemap,
    typing_info,
    func_ir,
    dist_analysis,
):
    """remove dead table columns using liveness info."""
    # We return True if any changes were made that could
    # allow for dead code elimination to make changes
    removed = False

    # List of tables that are updated at the source in this block.
    # add statements in reverse order
    new_body = [block.terminator]
    for stmt in reversed(block.body[:-1]):
        # Find all sources that create a table.
        if type(stmt) in remove_dead_column_extensions:
            f = remove_dead_column_extensions[type(stmt)]
            removed |= f(stmt, lives, equiv_vars, typemap)

        elif is_assign(stmt):
            lhs_name = stmt.target.name
            # Key when the output of the Operation is a table.
            lhs_table_key = (lhs_name, None)
            rhs = stmt.value
            if isinstance(stmt.value, ir.Expr) and stmt.value.op == "call":
                fdef = guard(numba.core.ir_utils.find_callname, func_ir, rhs)
                if fdef == (
                    "generate_mappable_table_func",
                    "bodo.utils.table_utils",
                ):
                    # In this case, if only a subset of the columns are live out of this mapped table function,
                    # we can pass this subset of columns to generate_mappable_table_func, which will allow generate_mappable_table_func
                    # To ignore these columns when mapping the function onto the table.
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )

                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue
                    nodes = compile_func_single_block(
                        eval(
                            "lambda table, func_name, out_arr_typ, is_method: bodo.utils.table_utils.generate_mappable_table_func(table, func_name, out_arr_typ, is_method, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue
                elif fdef in (
                    ("set_table_data", "bodo.hiframes.table"),
                    ("set_table_data_null", "bodo.hiframes.table"),
                ):
                    # Determine what columns are used by the output table. If we
                    # don't use the column being added/replaced we can potentially
                    # remove the column.
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )

                    if used_columns is None:
                        # if used_columns_for_current_table is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue
                    set_col_num = get_overload_const_int(typemap[rhs.args[1].name])
                    fname = fdef[0]
                    if set_col_num not in used_columns and fname == "set_table_data":
                        # Remove the use of the array if set_table_data column
                        # is not used.
                        nodes = compile_func_single_block(
                            eval(
                                "lambda table, idx: bodo.hiframes.table.set_table_data_null(table, idx, arr_typ, used_cols=used_columns)"
                            ),
                            [
                                rhs.args[0],
                                rhs.args[1],
                            ],
                            stmt.target,
                            typing_info=typing_info,
                            extra_globals={
                                "arr_typ": typemap[rhs.args[2].name],
                                "used_columns": bodo.utils.typing.MetaType(
                                    tuple(sorted(used_columns))
                                ),
                            },
                        )
                        removed = True
                    else:
                        # Remove set_col_num if present because it isn't needed.
                        used_columns.discard(set_col_num)
                        # Pass the used_cols to the function to skip loops, but
                        # keep the same function.
                        nodes = compile_func_single_block(
                            eval(
                                f"lambda table, idx, arr: bodo.hiframes.table.{fname}(table, idx, arr, used_cols=used_columns)"
                            ),
                            rhs.args,
                            stmt.target,
                            typing_info=typing_info,
                            extra_globals={
                                "used_columns": bodo.utils.typing.MetaType(
                                    tuple(sorted(used_columns))
                                )
                            },
                        )

                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    continue
                elif fdef == ("table_subset", "bodo.hiframes.table"):
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )
                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue

                    nodes = compile_func_single_block(
                        eval(
                            "lambda table, idx, copy_arrs: bodo.hiframes.table.table_subset(table, idx, copy_arrs, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )

                    # Replace the variable in the return value to keep
                    # distributed analysis consistent.
                    nodes[-1].target = stmt.target
                    # Update distributed analysis for the replaced function
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue

                elif fdef == ("logical_table_to_table", "bodo.hiframes.table"):
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )
                    # None means all columns are used
                    if used_columns is None:
                        new_body.append(stmt)
                        continue

                    kws = dict(rhs.kws)

                    if "used_cols" in kws:
                        prev_used_cols = set(
                            typemap[kws["used_cols"].name].instance_type.meta
                        )
                        # no need to update if used cols aren't changing
                        # otherwise will get stuck in a loop since we set removed=True
                        if prev_used_cols == used_columns:
                            new_body.append(stmt)
                            continue

                    # set dead input arguments to None to enable dead code elimination
                    in_col_inds = typemap[rhs.args[2].name].instance_type.meta
                    in_used_cols = {
                        in_ind
                        for out_ind, in_ind in enumerate(in_col_inds)
                        if out_ind in used_columns
                    }
                    n_in_table_arrs = get_overload_const_int(typemap[rhs.args[3].name])

                    # Always use the table because we can prune all of the columns.
                    in_table = "table"
                    n_extra_arrs = len(typemap[rhs.args[1].name].types)
                    if all(
                        i in in_used_cols
                        for i in range(n_in_table_arrs, n_in_table_arrs + n_extra_arrs)
                    ):
                        in_arrs = "arrs"
                    else:
                        arr_tup = ", ".join(
                            f"arrs[{i}]"
                            if i + n_in_table_arrs in in_used_cols
                            else "None"
                            for i in range(n_extra_arrs)
                        )
                        in_arrs = f"({arr_tup},)"
                    nodes = compile_func_single_block(
                        eval(
                            f"lambda table, arrs, col_inds, n_cols: bodo.hiframes.table.logical_table_to_table({in_table}, {in_arrs}, col_inds, n_cols, out_table_type_t=out_type, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            ),
                            "out_type": typemap[lhs_name],
                        },
                    )

                    # Replace the variable in the return value to keep
                    # distributed analysis consistent.
                    nodes[-1].target = stmt.target
                    # Update distributed analysis for the replaced function
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    removed = True
                    continue

                elif fdef == (
                    "table_astype",
                    "bodo.utils.table_utils",
                ):
                    # In this case, if only a subset of the columns are live
                    # we can skip converting the other columns
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )
                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue
                    nodes = compile_func_single_block(
                        eval(
                            "lambda table, new_table_typ, copy, _bodo_nan_to_str: bodo.utils.table_utils.table_astype(table, new_table_typ, copy, _bodo_nan_to_str, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue
                elif fdef == ("table_filter", "bodo.hiframes.table"):
                    # In this case, we've encountered a getitem that filters
                    # the rows of a table. At this step, we can also
                    # filter out columns that are not live out of this statement.

                    # Compute all columns that are live at this statement.
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )
                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue

                    nodes = compile_func_single_block(
                        eval(
                            "lambda table, idx: bodo.hiframes.table.table_filter(table, idx, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )

                    # Replace the variable in the return value to keep
                    # distributed analysis consistent.
                    nodes[-1].target = stmt.target
                    # Update distributed analysis for the replaced function
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue
                elif fdef == ("table_local_filter", "bodo.hiframes.table"):
                    # In this case, we've encountered a getitem that filters
                    # the rows of a table. At this step, we can also
                    # filter out columns that are not live out of this statement.

                    # Compute all columns that are live at this statement.
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )
                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue

                    nodes = compile_func_single_block(
                        eval(
                            "lambda table, idx: bodo.hiframes.table.table_local_filter(table, idx, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )

                    # Replace the variable in the return value to keep
                    # distributed analysis consistent.
                    nodes[-1].target = stmt.target
                    # Update distributed analysis for the replaced function
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue
                elif fdef == ("concat_tables", "bodo.utils.table_utils"):
                    # Compute all columns that are live at this statement.
                    used_columns = _find_used_columns(
                        lhs_table_key,
                        len(typemap[lhs_name].arr_types),
                        lives,
                        equiv_vars,
                    )
                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue

                    nodes = compile_func_single_block(
                        eval(
                            "lambda in_tables: bodo.utils.table_utils.concat_tables(in_tables, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )

                    # Replace the variable in the return value to keep
                    # distributed analysis consistent.
                    nodes[-1].target = stmt.target
                    # Update distributed analysis for the replaced function
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue
                elif fdef == ("read_arrow_next", "bodo.io.arrow_reader"):
                    # Compute all columns that are live at this statement
                    table_key = (lhs_name, 0)
                    used_columns = _find_used_columns(
                        table_key,
                        len(typemap[lhs_name][0].arr_types),
                        lives,
                        equiv_vars,
                    )
                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue

                    nodes = compile_func_single_block(
                        eval(
                            "lambda arrow_reader, produce_output: bodo.io.arrow_reader.read_arrow_next(arrow_reader, produce_output, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )

                    # Replace the variable in the return value to keep
                    # distributed analysis consistent.
                    nodes[-1].target = stmt.target
                    # Update distributed analysis for the replaced function
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue
                elif fdef == (
                    "join_probe_consume_batch",
                    "bodo.libs.streaming.join",
                ):
                    # Prune any unused columns from the output of join_probe_consume_batch.
                    # This avoids outputting unnecessary keys.

                    table_key = (lhs_name, 0)
                    used_columns = _find_used_columns(
                        table_key,
                        len(typemap[lhs_name][0].arr_types),
                        lives,
                        equiv_vars,
                    )
                    if used_columns is None:
                        # if used_columns is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue

                    nodes = compile_func_single_block(
                        eval(
                            "lambda join_state, table, is_last, produce_output: bodo.libs.streaming.join.join_probe_consume_batch(join_state, table, is_last, produce_output, used_cols=used_columns)"
                        ),
                        rhs.args,
                        stmt.target,
                        typing_info=typing_info,
                        extra_globals={
                            "used_columns": bodo.utils.typing.MetaType(
                                tuple(sorted(used_columns))
                            )
                        },
                    )

                    # Replace the variable in the return value to keep
                    # distributed analysis consistent.
                    nodes[-1].target = stmt.target
                    # Update distributed analysis for the replaced function
                    new_nodes = list(reversed(nodes))
                    if dist_analysis:
                        bodo.transforms.distributed_analysis.propagate_assign(
                            dist_analysis.array_dists, new_nodes
                        )
                    new_body += new_nodes
                    # We do not set removed = True here, as this branch does not make
                    # any changes that could allow removal in dead code elimination.
                    continue

        new_body.append(stmt)

    new_body.reverse()
    block.body = new_body
    return removed


def remove_dead_table_columns(
    func_ir,
    typemap,
    typing_info,
    dist_analysis=None,
):
    """
    Runs table liveness analysis and eliminates columns from TableType
    creation functions. This must be run before custom IR extensions are
    transformed. The function returns True if any changes were made that could
    allow for dead code elimination to make changes.
    """
    # We return True if any changes were made that could
    # allow for dead code elimination to make changes
    removed = False
    # Only run remove_dead_columns if some table exists.
    run_dead_elim = False
    for typ in typemap.values():
        if isinstance(typ, TableType):
            run_dead_elim = True
            break
    if run_dead_elim:
        blocks = func_ir.blocks
        cfg = compute_cfg_from_blocks(blocks)
        column_live_map, column_equiv_vars = compute_column_liveness(
            cfg, blocks, func_ir, typemap
        )
        for label, block in blocks.items():
            removed |= remove_dead_columns(
                block,
                column_live_map[label],
                column_equiv_vars,
                typemap,
                typing_info,
                func_ir,
                dist_analysis,
            )
    func_ir._definitions = build_definitions(func_ir.blocks)
    return removed


def get_live_column_nums_block(block_lives, equiv_vars, table_key):
    """Given a finalized live map for a block, computes the actual
    column numbers that are used by the table. For efficiency this returns
    two values, a sorted list of column numbers and a use_all flag.
    If use_all=True the column numbers are garbage."""
    total_used_columns, use_all, cannot_del_cols = block_lives.get(
        table_key, (set(), False, False)
    )
    if use_all or cannot_del_cols:
        return set(), use_all, cannot_del_cols
    aliases = equiv_vars[table_key]
    for var_key in aliases:
        new_columns, use_all, cannot_del_cols = block_lives.get(
            var_key, (set(), False, False)
        )
        if use_all or cannot_del_cols:
            return set(), use_all, cannot_del_cols
        total_used_columns = total_used_columns | new_columns
    return total_used_columns, False, False


def _find_used_columns(lhs_key, max_num_cols, lives, equiv_vars):
    """
    Finds the used columns needed at a particular block.
    This is used for functions that update the code to include
    a "used_cols" in an optimization path.

    Returns None if all columns are used, otherwise a set
    with the used columns.
    """
    # Compute all columns that are live at this statement.
    used_columns, use_all, cannot_del_cols = get_live_column_nums_block(
        lives, equiv_vars, lhs_key
    )
    if use_all or cannot_del_cols:
        return None
    used_columns = bodo.ir.connector.trim_extra_used_columns(used_columns, max_num_cols)
    return used_columns


def _generate_rhs_use_map(
    rhs_key, block_use_map, table_col_use_map, equiv_vars, lhs_key
):
    """
    Finds the uses from an operation that makes copies of the lists (so
    it is not an alias), but all uses from the lhs table are uses for
    the rhs table. (e.g. filter). An operation uses this when it doesn't
    directly add any additional column uses and all column uses are a
    function of any uses of its output table and any of its aliases.

    Returns a triple of values:
        - used_columns: set of used columns
        - use_all (boolean): If true, used_columns will be the empty set.
        - cannot_del_cols (boolean): If true we cannot prune
          any columns for this table or any alias.
    """
    used_columns, use_all, cannot_del_cols = block_use_map[rhs_key]
    if use_all or cannot_del_cols:
        return set(), use_all, cannot_del_cols
    lhs_used_columns, lhs_use_all, lhs_cannot_del_cols = _compute_table_column_uses(
        lhs_key, table_col_use_map, equiv_vars
    )
    if lhs_use_all or lhs_cannot_del_cols:
        # All map operations make a copy of the underlying
        # table. As a result, every column is needed for this
        # operation, but the columns can be safely deleted
        # from the prior table.
        return set(), True, False
    return used_columns | lhs_used_columns, False, False


def _compute_table_column_uses(
    table_key: tuple[str, int | None],
    table_col_use_map: dict[
        int, dict[tuple[str, int | None], tuple[set[int], bool, bool]]
    ],
    equiv_vars: pt.Mapping[str, set[str]],
):
    """Computes the used columns for a table name
    and all of its aliases. Returns a triple of
    used_columns, use_all (a flag to avoid passing
    large used_columns sets), and cannot_del_cols,
    a flag that indicates we should not delete any
    columns.
    Args:
        table_name (Tuple[str, Optional[int]]): The to find the table and to determine uses.
        table_col_use_map (dict[int, dict[str, tuple[set[int], bool, bool]]]):
            Dictionary mapping block numbers to a dictionary of table keys
            and "column uses". A column use is represented by the triple
                - used_cols: Set of used column numbers
                - use_all: Flag for if all columns are used. If True used_cols
                    is garbage
                - cannot_del_columns: Flag indicate this table is used in
                    an unsupported operation (e.g. passed to a DataFrame)
                    and therefore no columns can be deleted.
        equiv_vars (Mapping[str, set[str]]): Dictionary
            mapping table variable names to a set of
            other table name aliases.
    Returns:
        tuple[set[int], bool, bool]: Returns a triple of column uses
            for the table in the current A column use is represented
            by the triple
                - used_cols: Set of used column numbers
                - use_all: Flag for if all columns are used. If True used_cols
                    is garbage
                - cannot_del_columns: Flag indicate this table is used in
                    an unsupported operation (e.g. passed to a DataFrame)
                    and therefore no columns can be deleted.
    """
    used_columns = set()
    for other_block_use_map in table_col_use_map.values():
        used_col_local, use_all, cannot_del_cols = get_live_column_nums_block(
            other_block_use_map, equiv_vars, table_key
        )
        if use_all or cannot_del_cols:
            return set(), use_all, cannot_del_cols
        used_columns.update(used_col_local)
    return used_columns, False, False


def _update_equiv_set(equiv_vars, lhs_key, rhs_key):
    """Update equivalent variable sets for lhs and rhs to set them as equivalent"""
    if lhs_key not in equiv_vars:
        equiv_vars[lhs_key] = set()
    if rhs_key not in equiv_vars:
        equiv_vars[rhs_key] = set()
    equiv_vars[lhs_key].add(rhs_key)
    equiv_vars[rhs_key].add(lhs_key)
    equiv_vars[lhs_key] |= equiv_vars[rhs_key]
    equiv_vars[rhs_key] |= equiv_vars[lhs_key]
