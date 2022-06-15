# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Updates the function IR to include decref on individual columns
when they are no longer used. This enables garbage collecting
single columns when tables are represented by a single variable.
"""
import copy
import functools
import operator
from collections import defaultdict

import numba
import numpy as np
from numba.core import ir, types
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.ir_utils import build_definitions, find_topo_order, guard

import bodo
from bodo.hiframes.table import TableType, del_column, gen_table_filter
from bodo.utils.transform import compile_func_single_block
from bodo.utils.typing import get_overload_const_int, is_list_like_index_type
from bodo.utils.utils import is_assign, is_expr


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
            cfg, f_ir.blocks, f_ir, typemap, False
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

            'alias_sets':    A dictionary of unique alias sets (table_name -> aliases)

            'liveness_map':  A dictionary mapping each block to a
                             dictionary {table_name -> set(live_columns)}
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
        # The source node will contain which variables use all columns, which
        # we must omit.
        source_live_cols = column_live_map[cfg.entry_point()]
        alias_sets = _find_alias_sets(source_live_cols, equiv_vars)
        # {block_offset -> {table_name: set(used_columns)}}
        # The alias_sets and table_name in livecol_map share a common representative
        # variable.
        livecol_map = {}
        for offset in blocks.keys():
            block_column_liveness = column_live_map[offset]
            # Group live columns within an alias set.
            livecol_map[offset] = set_dict = {}
            for table_name in alias_sets.keys():
                used_columns, _ = get_live_column_nums_block(
                    block_column_liveness, equiv_vars, table_name
                )
                # Convert back to a set for easier comparison between blocks.
                set_dict[table_name] = set(used_columns)

        return alias_sets, livecol_map

    def compute_column_del_locations(
        self, cfg, blocks, alias_sets, alias_set_livecol_map
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
            for table_name, aliases in alias_sets.items():
                # Determine the variables to delete in this block
                curr_cols = block_livecols[table_name]
                # Find all columns that are live in any successor
                succ_livecols = [
                    alias_set_livecol_map[label][table_name]
                    for label, _ in cfg.successors(offset)
                ]
                combined_succ_livecols = functools.reduce(
                    operator.or_, succ_livecols, set()
                )
                # Delete columns that aren't live in any successor.
                del_curr = curr_cols - combined_succ_livecols
                col_back_map[table_name] = del_curr
                escaping_live_set = curr_cols - del_curr
                for label, _ in cfg.successors(offset):
                    # For each successor delete columns that only need
                    # to be deleted from that column
                    escaping_dead[label][table_name] |= (
                        escaping_live_set - alias_set_livecol_map[label][table_name]
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
        we find each get_dataframe_data and insert the del immediately afterwards.

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
                func_text = f"def del_columns({args}):\n"
                for table_name, columns in escaping_cols.items():
                    used_var_name = get_livevar_name(
                        table_name, alias_sets[table_name], block_livemap
                    )
                    if used_var_name and columns:
                        # We only add this table if there is at least 1 live
                        # var and columns is not empty. See test_table_dead_var
                        updated = True
                        var_names.append(used_var_name)
                        for column in sorted(columns):
                            # Decref each column to remove.
                            func_text += f"    del_column(arg{ctr}, {column})\n"
                        ctr += 1
                if var_names:
                    # Only compile the function if at least 1 table needs to be deleted.
                    new_stmts = self._compile_del_column_function(func_text, var_names)
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
                        if fdef == ("get_table_data", "bodo.hiframes.table"):
                            table_var_name = rhs.args[0].name
                            col_num = typemap[rhs.args[1].name].literal_value
                            # Determine the table key.
                            if table_var_name in internal_cols:
                                cols = internal_cols[table_var_name]
                            else:
                                # If the name we found isn't the representative
                                # for the alias, find the representative if it exists.
                                cols = set()
                                s = equiv_vars[table_var_name]
                                for table_name in sorted(s):
                                    if table_name in internal_cols:
                                        cols = internal_cols[table_name]
                                        break
                            # If we have already removed this column ignore it.
                            if col_num in cols:
                                updated = True
                                func_text = f"def del_columns(table_arg):\n"
                                func_text += f"    del_column(table_arg, {col_num})\n"
                                # Compile the function
                                new_stmts = self._compile_del_column_function(
                                    func_text, [table_var_name]
                                )
                                # Insert stmts in reverse order because we reverse the block
                                new_body.extend(list(reversed(new_stmts)))
                                # Mark the column as removed
                                cols.remove(col_num)
                    new_body.append(stmt)
                # We insert into the body in reverse order, so reverse it again.
                new_body = list(reversed(new_body))
            block.body = insert_front + new_body
        return updated

    def _compile_del_column_function(self, func_text, var_names):
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
            extra_globals={
                "del_column": del_column,
            },
        )


def _find_alias_sets(source_live_cols, equiv_vars):
    """
    Given a live map for the source block and the equiv_vars dictionary,
    returns a dictionary of all unique alias sets, omitting any sets that
    use all columns throughout the program.
    """
    seen_vars = set()
    alias_sets = {}
    for table_name in source_live_cols.keys():
        if table_name not in seen_vars:
            # Mark all members of the alias set as seen
            seen_vars.add(table_name)
            seen_vars.update(equiv_vars[table_name])
            _, use_all = get_live_column_nums_block(
                source_live_cols, equiv_vars, table_name
            )
            if not use_all:
                # Only add the set if it doesn't use all column
                alias_sets[table_name] = equiv_vars[table_name]
    return alias_sets


# Global dictionaries for IR extensions to handle dead column analysis
remove_dead_column_extensions = {}
ir_extension_table_column_use = {}


def get_livevar_name(table_name, aliases, block_livemap):
    """
    Given a set of alias names, return the name
    of a variable live in the block_livemap or ""
    if no variable is found.
    """
    # Sort to ensure this is with multiple ranks.
    if table_name in block_livemap:
        return table_name
    for alias in sorted(aliases):
        if alias in block_livemap:
            return alias
    return ""


def compute_column_liveness(cfg, blocks, func_ir, typemap, trim_aliases):
    """
    Compute which columns may be live in each block. These are computed
    according to the DataFlow algorithm found in this confluence document

    https://bodo.atlassian.net/wiki/spaces/B/pages/912949249/Dead+Column+Elimination+with+Large+Numbers+of+Columns

    We keep the use and live computations as closures to make these functions "private"
    to this function, which should always be called instead.
    """
    column_usemap, equiv_vars = _compute_table_column_use(blocks, func_ir, typemap)
    column_live_map = _compute_table_column_live_map(
        cfg, blocks, typemap, column_usemap
    )
    return column_live_map, equiv_vars


def _compute_table_column_use(blocks, func_ir, typemap):
    """
    Find table column use per block. This returns two dictionaries,
    one that tracks columns used for each block, 'table_col_use_map'
    and a dictionary that tracks aliases 'equiv_vars'.

    'table_col_use_map' values are a tuple of 2 values:

        1. A set of columns directly used.
        2. A boolean flag for if all column are used. This is an optimization
        to reduce memory when there are operations that don't support
        removing tables.

    'equiv_vars' maps each table name to all other tables that function as an alias.
    """
    # keep track of potential aliases for tables
    equiv_vars = defaultdict(set)
    table_col_use_map = (
        {}
    )  # { block offset -> dict(var_name -> tuple(set(used_column_numbers), use_all)}
    # See table filter note below regarding reverse order
    for offset in reversed(find_topo_order(blocks)):
        ir_block = blocks[offset]
        table_col_use_map[offset] = block_use_map = defaultdict(lambda: (set(), False))
        for stmt in reversed(ir_block.body):

            # IR extensions that impact column uses.
            if type(stmt) in ir_extension_table_column_use:
                f = ir_extension_table_column_use[type(stmt)]
                # We assume that f checks types if necessary
                # and performs any required actions.
                f(stmt, block_use_map, equiv_vars, typemap)
                continue

            # If we have an assign we don't want to mark that variable as used.
            lhs_name = None

            # Check for assigns. This should include all basic usages
            # (get_dataframe_data + simple assign)
            if isinstance(stmt, ir.Assign):
                rhs = stmt.value
                lhs_name = stmt.target.name
                # Handle simple assignments (i.e. df2 = df1)
                # This should add update the equiv_vars
                if isinstance(stmt.value, ir.Var) and isinstance(
                    typemap[lhs_name], TableType
                ):
                    rhs_name = stmt.value.name
                    if lhs_name not in equiv_vars:
                        equiv_vars[lhs_name] = set()
                    if rhs_name not in equiv_vars:
                        equiv_vars[rhs_name] = set()
                    equiv_vars[lhs_name].add(rhs_name)
                    equiv_vars[rhs_name].add(lhs_name)
                    equiv_vars[lhs_name] |= equiv_vars[rhs_name]
                    equiv_vars[rhs_name] |= equiv_vars[lhs_name]
                    continue
                # Handle calls to get_table_data
                elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "call":
                    rhs = stmt.value
                    fdef = guard(numba.core.ir_utils.find_callname, func_ir, rhs)
                    if fdef == ("get_table_data", "bodo.hiframes.table"):
                        table_var_name = rhs.args[0].name
                        assert isinstance(
                            typemap[table_var_name], TableType
                        ), "Internal Error: Invalid get_table_data call"
                        col_num = typemap[rhs.args[1].name].literal_value
                        col_num_set = block_use_map[table_var_name][0]
                        col_num_set.add(col_num)
                        continue
                    elif fdef in (
                        ("set_table_data", "bodo.hiframes.table"),
                        (
                            "set_table_data_null",
                            "bodo.hiframes.table",
                        ),
                    ):
                        # set_table_data and set_table_data_null both reuse lists from the input table.
                        # IE, the table rhs table cannot be reused anywhere in the CFG after the setitem
                        # Therefore, for the purposes of determing column uses/liveness,
                        # the lhs table is an alias of the rhs table, and the rhs table needs to consider
                        # the lhs's uses, but the rhs table is not an
                        # alias of the lhs table, and ths lhs does not need to consider the rhs's uses.
                        rhs_name = rhs.args[0].name
                        assert isinstance(
                            typemap[rhs_name], TableType
                        ), f"Internal Error: Invalid {fdef[0]} call"
                        if lhs_name not in equiv_vars:
                            equiv_vars[lhs_name] = set()
                        if rhs_name not in equiv_vars:
                            equiv_vars[rhs_name] = set()
                        equiv_vars[rhs_name].add(lhs_name)
                        equiv_vars[rhs_name] |= equiv_vars[lhs_name]
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
                        used_cols, use_all = _generate_rhs_use_effective_alias(
                            rhs_table,
                            block_use_map,
                            table_col_use_map,
                            equiv_vars,
                            lhs_name,
                        )
                        block_use_map[rhs_table] = (used_cols, use_all)
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
                        used_cols, use_all = _generate_rhs_use_effective_alias(
                            rhs_table,
                            block_use_map,
                            table_col_use_map,
                            equiv_vars,
                            lhs_name,
                        )
                        block_use_map[rhs_table] = (used_cols, use_all)
                        continue
                elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getattr":
                    if stmt.value.attr == "shape":
                        # Skip ops that shouldn't impact the number of columns. Shape
                        # can be computed from a combination of the tyep and the length
                        # of any column, but no particular column. This needs to be skipped
                        # because it is inserted by
                        continue

                # handle table filter like T2 = T1[ind]
                elif (
                    is_expr(rhs, "getitem")
                    and isinstance(typemap[rhs.value.name], TableType)
                    and (
                        (
                            is_list_like_index_type(typemap[rhs.index.name])
                            and typemap[rhs.index.name].dtype == types.bool_
                        )
                        or isinstance(typemap[rhs.index.name], types.SliceType)
                    )
                ):
                    # NOTE: column uses of input T1 are the same as output T2.
                    # Here we simply traverse the IR in reversed order and update uses,
                    # which works because table filter variables are internally
                    # generated variables and have a single definition without control
                    # flow. Otherwise, we'd need to update uses iteratively.

                    # NOTE: We must search the entire table_col_use_map at this point
                    # because we haven't updated column usage/use_all from successor
                    # blocks yet.

                    rhs_table = rhs.value.name
                    used_cols, use_all = _generate_rhs_use_effective_alias(
                        rhs_table,
                        block_use_map,
                        table_col_use_map,
                        equiv_vars,
                        lhs_name,
                    )
                    block_use_map[rhs_table] = (used_cols, use_all)
                    continue

            for var in stmt.list_vars():
                # All unknown table uses should use all columns.
                if var.name != lhs_name and isinstance(typemap[var.name], TableType):
                    # If a statement is used in any ordinary way (i.e returned)
                    # then we mark all columns as used. We use a boolean as a shortcut.
                    block_use_map[var.name] = (block_use_map[var.name][0], True)

    return table_col_use_map, equiv_vars


def _compute_table_column_live_map(cfg, blocks, typemap, column_uses):
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
        We detect this by determining the column numbers, use_all_flag
        values haven't changed.
        """
        results = []
        for vals in dct.values():
            results.append(tuple((len(v[0]), v[1]) for v in vals.values()))
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
            for inc_blk, _data in cfg.predecessors(offset):
                for df_name, liveness_tup in live_columns.items():
                    # Initialize the df if it doesn't exist or if liveness_tup[1]=True.
                    if df_name not in dct[inc_blk]:
                        dct[inc_blk][df_name] = (
                            liveness_tup[0].copy(),
                            liveness_tup[1],
                        )
                    else:
                        pred_liveness_tup = dct[inc_blk][df_name]
                        if liveness_tup[1] or pred_liveness_tup[1]:
                            # If either use_all is true we can
                            # remove any column numbers to save memory.
                            dct[inc_blk][df_name] = (set(), True)
                        else:
                            dct[inc_blk][df_name] = (
                                pred_liveness_tup[0] | liveness_tup[0],
                                False,
                            )

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
    allow_liveness_breaking_changes,
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
            rhs = stmt.value
            if allow_liveness_breaking_changes and (
                is_expr(rhs, "getitem")
                and isinstance(typemap[rhs.value.name], TableType)
                and (
                    (
                        is_list_like_index_type(typemap[rhs.index.name])
                        and typemap[rhs.index.name].dtype == types.bool_
                    )
                    or isinstance(typemap[rhs.index.name], types.SliceType)
                )
            ):
                # In this case, we've encountered a getitem that filters
                # the rows of a table. At this step, we can also
                # filter out columns that are not live out of this statment.

                # Note that we are replaceing this getitem with a custom generated filter function.
                # Therefore, on subsequent passes, liveness analysis will only see a call to a function,
                # and conservativly assume that all columns in the argument table are live. This will result
                # in reduced liveness information, which impacts our ability to perform other optimizations.
                # Therefore, we only perform this optimization on the last pass of table column elimination,
                # when allow_liveness_breaking_changes is True.
                # TODO: fix this workaround for better clarity, see
                # https://bodo.atlassian.net/jira/software/projects/BE/boards/4/backlog?selectedIssue=BE-3033

                # Compute all columns that are live at this statement.
                used_columns = _find_used_columns(
                    lhs_name, typemap[rhs.value.name], lives, equiv_vars, typemap
                )
                if used_columns is None:
                    # if used_columns is None it means all columns are used.
                    # As such, we can't do any column pruning
                    new_body.append(stmt)
                    continue
                filter_func = gen_table_filter(typemap[rhs.value.name], used_columns)
                filter_func = numba.njit(filter_func, no_cpython_wrapper=True)
                nodes = compile_func_single_block(
                    eval("lambda T, ind: _filter_func(T, ind)"),
                    (rhs.value, rhs.index),
                    ret_var=None,
                    typing_info=typing_info,
                    extra_globals={"_filter_func": filter_func},
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
            elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "call":
                fdef = guard(numba.core.ir_utils.find_callname, func_ir, rhs)
                if allow_liveness_breaking_changes and fdef == (
                    "generate_mappable_table_func",
                    "bodo.utils.table_utils",
                ):
                    # In this case, if only a subset of the columns are live out of this maped table function,
                    # we can pass this subset of columns to generate_mappable_table_func, which will allow generate_mappable_table_func
                    # To ignore these columns when mapping the function onto the table.
                    used_columns = _find_used_columns(
                        lhs_name, typemap[rhs.args[0].name], lives, equiv_vars, typemap
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
                        extra_globals={"used_columns": used_columns},
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
                elif fdef == ("set_table_data", "bodo.hiframes.table"):
                    # In this case, we have a set_table_data statement, where the set column is not used.
                    # We can replace set_table_data with set_table_data_null, which is an equivalent statment that doesn't use the array
                    # but still changes the type of the table. This will potentially allow for dead code elimination to do work

                    # NOTE: In this case, we check the left hand table, because set_table_data can add new columns,
                    # and checking the right hand column would exclude any newly added columns from the used_columns list.

                    used_columns = _find_used_columns(
                        lhs_name, typemap[lhs_name], lives, equiv_vars, typemap
                    )

                    if used_columns is None:
                        # if used_columns_for_current_table is None it means all columns are used.
                        # As such, we can't do any column pruning
                        new_body.append(stmt)
                        continue
                    set_col_num = get_overload_const_int(typemap[rhs.args[1].name])
                    if set_col_num not in used_columns:
                        nodes = compile_func_single_block(
                            eval(
                                "lambda table, idx: bodo.hiframes.table.set_table_data_null(table, idx, arr_typ)"
                            ),
                            [
                                rhs.args[0],
                                rhs.args[1],
                            ],
                            stmt.target,
                            typing_info=typing_info,
                            extra_globals={"arr_typ": typemap[rhs.args[2].name]},
                        )
                        new_nodes = list(reversed(nodes))
                        if dist_analysis:
                            bodo.transforms.distributed_analysis.propagate_assign(
                                dist_analysis.array_dists, new_nodes
                            )
                        new_body += new_nodes
                        removed = True
                        continue
                    else:
                        new_body.append(stmt)
                        continue
                elif allow_liveness_breaking_changes and fdef == (
                    "table_astype",
                    "bodo.utils.table_utils",
                ):
                    # In this case, if only a subset of the columns are live
                    # we can skip converting the other columns
                    used_columns = _find_used_columns(
                        lhs_name, typemap[lhs_name], lives, equiv_vars, typemap
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
                        extra_globals={"used_columns": used_columns},
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

        new_body.append(stmt)

    new_body.reverse()
    block.body = new_body
    return removed


def remove_dead_table_columns(
    func_ir,
    typemap,
    typing_info,
    dist_analysis=None,
    allow_liveness_breaking_changes=True,
):
    """
    Runs table liveness analysis and eliminates columns from TableType
    creation functions. This must be run before custom IR extensions are
    transformed. Thie function returns if any changes were made that could
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
            cfg, blocks, func_ir, typemap, True
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
                allow_liveness_breaking_changes,
            )
    func_ir._definitions = build_definitions(func_ir.blocks)
    return removed


def get_live_column_nums_block(block_lives, equiv_vars, table_name):
    """Given a finalized live map for a block, computes the actual
    column numbers that are used by the table. For efficiency this returns
    two values, a sorted list of column numbers and a use_all flag.
    If use_all=True the column numbers are garbage."""
    total_used_columns, use_all = block_lives.get(table_name, (set(), False))
    if use_all:
        return [], True
    aliases = equiv_vars[table_name]
    for var_name in aliases:
        new_columns, use_all = block_lives.get(var_name, (set(), False))
        if use_all:
            return [], True
        total_used_columns = total_used_columns | new_columns
    return sorted(total_used_columns), False


def _find_used_columns(lhs_name, table_type, lives, equiv_vars, typemap):
    """
    Finds the used columns needed at a particular block.
    This is used for functions that update the code to include
    a "used_cols" in an optimization path.

    Returns None if all columns are used, otherwise an np.array
    with the used columns.
    """
    # Compute all columns that are live at this statement.
    used_columns, use_all = get_live_column_nums_block(lives, equiv_vars, lhs_name)
    if use_all:
        return None
    used_columns = bodo.ir.connector.trim_extra_used_columns(
        used_columns, len(table_type.arr_types)
    )
    return np.array(used_columns, dtype=np.int64)


def _generate_rhs_use_effective_alias(
    rhs_table, block_use_map, table_col_use_map, equiv_vars, lhs_name
):
    """
    Finds the uses from an operation that effectively acts like an alias
    (e.g. filter). An operation acts like an alias when it doesn't directly
    add any additional column uses and all column uses are a function of any
    uses of its output.

    Returns a pair of values:
        - used_columns: set of used columns
        - use_all: Boolean. If true, used_columns will be the empty set.
    """
    used_columns, use_all = block_use_map[rhs_table]
    if use_all:
        return set(), True
    # Make a copy in case the set is shared.
    used_columns = used_columns.copy()
    for other_block_use_map in table_col_use_map.values():
        used_col_local, use_all = get_live_column_nums_block(
            other_block_use_map, equiv_vars, lhs_name
        )
        if use_all:
            return set(), True
        used_columns.update(used_col_local)
    return used_columns, False
