"""
transforms the IR to handle bytecode issues in Python 3.10.
"""

import operator

import numba
from numba.core import ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import dprint_func_ir, get_definition, guard


@register_pass(mutates_CFG=False, analysis_only=False)
class Bodo310ByteCodePass(FunctionPass):
    _name = "bodo_untyped_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Fix IR before typing to handle untypeable cases
        """
        # Ensure we have an IR.
        assert state.func_ir
        dprint_func_ir(state.func_ir, "starting Bodo 3.10 Bytecode optimizations pass")
        peep_hole_fuse_dict_add_updates(state.func_ir)
        peep_hole_fuse_tuple_adds(state.func_ir)
        peep_hole_fix_build_list(state.func_ir)
        return True


def peep_hole_fuse_tuple_adds(func_ir):
    """
    This rewrite removes t3 = t1 + t2 exprs that
    are between two tuples, t1 and t2 in the
    same basic block, resulting from a Python 3.10
    upgrade. If both expressions are build tuples
    defined in that block and neither is used between
    the add call, then we replace t3 with a new definition
    that combines the two tuples into a single build_tuple.

    At this time we cannot differentiate between user code
    and bytecode generated code.
    """

    # This algorithm fuses tuple add expressions into the largest
    # possible build_tuple before usage. For example, if we have an
    # IR that looks like this:
    #
    #   $t0 = build_tuple([])
    #   $val1 = const(2)
    #   $t1 = build_tuple([$val1])
    #   $append_t1_t0 = $t0 + $t1
    #   $val2 = const(2)
    #   $t2 = build_tuple([$val2])
    #   $append_t2_t1 = $t1 + $append_t1_t0
    #   $val3 = const(2)
    #   $t3 = build_tuple([$val3])
    #   $append_t3_t2 = $t2 + $append_t2_t1
    #   $val4 = const(2)
    #   $t4 = build_tuple([$val4])
    #   $append_t4_t3 = $t3 + $append_t3_t2
    #   $finalvar = $append_t4_t3
    #   $retvar = cast($finalvar)
    #   return $retvar
    #
    # It gets converted into
    #
    #   $t0 = build_tuple([])
    #   $val1 = const(2)
    #   $t1 = build_tuple([$val1])
    #   $append_t1_t0 = build_tuple([$val1])
    #   $val2 = const(2)
    #   $t2 = build_tuple([$val2])
    #   $append_t2_t1 = build_tuple([$val1, $val2])
    #   $val3 = const(2)
    #   $t3 = build_tuple([$val3])
    #   $append_t3_t2 = build_tuple([$val1, $val2, $val3])
    #   $val4 = const(2)
    #   $t4 = build_tuple([$val4])
    #   $append_t4_t3 = build_tuple([$val1, $val2, $val3, $val4])
    #   $finalvar = $append_t4_t3
    #   $retvar = cast($finalvar)
    #   return $retvar
    #
    # We then depend on the dead code elimination in untyped pass to remove
    # any unused tuple.

    for blk in func_ir.blocks.values():
        new_body = []
        # var name -> list of items for build tuple
        build_tuple_map = {}
        for i, stmt in enumerate(blk.body):
            stmt_build_tuple_out = None
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                lhs = stmt.target.name
                if stmt.value.op == "build_tuple":
                    # Add any build tuples to the list to track
                    stmt_build_tuple_out = lhs
                    build_tuple_map[lhs] = stmt.value.items
                elif (
                    stmt.value.op == "binop"
                    and stmt.value.fn == operator.add
                    and stmt.value.lhs.name in build_tuple_map
                    and stmt.value.rhs.name in build_tuple_map
                ):
                    stmt_build_tuple_out = lhs
                    # If we have an add between two build_tuples we are tracking we replace the tuple.
                    new_items = (
                        build_tuple_map[stmt.value.lhs.name]
                        + build_tuple_map[stmt.value.rhs.name]
                    )
                    new_build_tuple = ir.Expr.build_tuple(new_items, stmt.value.loc)
                    # Add the new tuple to track
                    build_tuple_map[lhs] = new_items
                    # Each tuple should be used only once.
                    del build_tuple_map[stmt.value.lhs.name]
                    del build_tuple_map[stmt.value.rhs.name]
                    # Delete the old definition
                    if stmt.value in func_ir._definitions[lhs]:
                        func_ir._definitions[lhs].remove(stmt.value)
                    # Add the new defintion
                    func_ir._definitions[lhs].append(new_build_tuple)
                    # Replace the stmt
                    stmt = ir.Assign(new_build_tuple, stmt.target, stmt.loc)

            for var in stmt.list_vars():
                # We only want to replace tuples that are unused
                # except for a single add. As result we delete any
                # tuples from the constant list once they are used.
                if var.name in build_tuple_map and var.name != stmt_build_tuple_out:
                    del build_tuple_map[var.name]
            new_body.append(stmt)

        blk.body = new_body
    return func_ir


# The code below is all copied from https://github.com/numba/numba/pull/7964


def peep_hole_fuse_dict_add_updates(func_ir):
    """
    This rewrite removes d1.update(d2) calls that
    are between two dictionaries, d1 and d2 in the
    same basic block, resulting from a Python 3.10
    upgrade. If both are constant dictionaries
    defined in that block and neither is used between
    the update call, then we replace d1 with a new definition
    that combines the two dictionaries.
    Python 3.10 may also rewrite a dictionary as an empty
    build_map + many map_add, so we also need to replace those
    expressions with a constant build map.
    """

    # This algorithm fuses build_map expressions into the largest
    # possible build map before usage. For example, if we have an
    # IR that looks like this:
    #
    #   $d1 = build_map([])
    #   $key = const("a")
    #   $value = const(2)
    #   $setitem_func = getattr($d1, "__setitem__")
    #   $unused1 = call (setitem_func, ($key, $value))
    #   $key2 = const("b")
    #   $value2 = const(3)
    #   $d2 = build_map([($key2, $value2)])
    #   $update_func = getattr($d1, "update")
    #   $unused2 = call ($update_func, ($d2,))
    #   $othervar = None
    #   $retvar = cast($othervar)
    #   return $retvar
    #
    # Then the IR is rewritten so any __setitem__ and update operations are fused into
    # the original buildmap. The new buildmap is then add to the last location where it
    # had previously had encountered a __setitem__, update, or build_map before any other uses.
    # The new IR would look like:
    #
    #   $key = const("a")
    #   $value = const(2)
    #   $key2 = const("b")
    #   $value2 = const(3)
    #   $d2 = build_map([($key2, $value2)])
    #   $d1 = build_map([($key, $value), ($key2, $value2)])
    #   $othervar = None
    #   $retvar = cast($othervar)
    #   return $retvar
    #
    # Notice how we don't push $d1 to the bottom of the block. This is because
    # some values may be found below this block (e.g pop_block) that are pattern
    # matched in other locations, such as objmode handling.

    for blk in func_ir.blocks.values():
        new_body = []
        # literal map var name -> index of build_map assign in the original block body
        lit_old_idx = {}
        # literal map var name -> index of build_map assign in the new block body
        lit_new_idx = {}
        # literal map var name -> list of key/value items for build map
        map_updates = {}
        blk_changed = False

        for i, stmt in enumerate(blk.body):
            # Should we add the current inst to the output
            append_inst = True
            # Name that shoud be skipped when looking at used
            # vars.
            stmt_build_map_out = None
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                if stmt.value.op == "build_map":
                    # Skip the output build_map when looks for uses.
                    stmt_build_map_out = stmt.target.name
                    # If we encounter a build map add it to the
                    # tracked build_maps.
                    lit_old_idx[stmt.target.name] = i
                    lit_new_idx[stmt.target.name] = i
                    map_updates[stmt.target.name] = stmt.value.items.copy()
                    append_inst = False
                elif stmt.value.op == "call" and i > 0:
                    # If we encounter a call we may need to replace
                    # the body
                    func_name = stmt.value.func.name
                    # If we have an update or a setitem
                    # it will be the previous expression.
                    getattr_stmt = blk.body[i - 1]
                    args = stmt.value.args
                    if (
                        isinstance(getattr_stmt, ir.Assign)
                        and getattr_stmt.target.name == func_name
                        and isinstance(getattr_stmt.value, ir.Expr)
                        and getattr_stmt.value.op == "getattr"
                        and getattr_stmt.value.value.name in lit_old_idx
                    ):
                        update_map_name = getattr_stmt.value.value.name
                        attr = getattr_stmt.value.attr
                        if attr == "__setitem__":
                            append_inst = False
                            # If we have a setitem, update the lists
                            map_updates[update_map_name].append(args)
                            # Remove the setitem
                            new_body[-1] = None

                        elif attr == "update" and args[0].name in lit_old_idx:
                            append_inst = False
                            # If we have an update and the arg is also
                            # a literal dictionary, fuse the lists.
                            map_updates[update_map_name].extend(
                                map_updates[args[0].name]
                            )
                            # Remove the update
                            new_body[-1] = None
                        if not append_inst:
                            # The output of __setitem__ and update is now always
                            # unused so we delete the IR stmtx.
                            # Update the new insert location
                            lit_new_idx[update_map_name] = i
                            # Drop the existing definition for this stmt.
                            func_ir._definitions[getattr_stmt.target.name].remove(
                                getattr_stmt.value
                            )

            # Check if we need to pop any dictionaries from being tracked.
            # Skip the setitem/update gettar that will be removed when
            # handling their call in the next iteration.
            if not (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Expr)
                and stmt.value.op == "getattr"
                and stmt.value.value.name in lit_old_idx
                and stmt.value.attr in ("__setitem__", "update")
            ):
                for var in stmt.list_vars():
                    # If a dictionary is used it cannot be pushed farther into
                    # the block. Skip the assign target.
                    if var.name in lit_old_idx and var.name != stmt_build_map_out:
                        _insert_build_map(
                            func_ir,
                            var.name,
                            blk.body,
                            new_body,
                            lit_old_idx,
                            lit_new_idx,
                            map_updates,
                        )
            if append_inst:
                new_body.append(stmt)
            else:
                # Drop the existing definition for this stmt.
                func_ir._definitions[stmt.target.name].remove(stmt.value)
                blk_changed = True
                # Append None so the number of instructions remains the same.
                new_body.append(None)

        # Insert any remaining maps. We make a list of keys because
        # we modify lit_old_idx in the loop.
        keys = list(lit_old_idx.keys())
        for var_name in keys:
            _insert_build_map(
                func_ir,
                var_name,
                blk.body,
                new_body,
                lit_old_idx,
                lit_new_idx,
                map_updates,
            )
        if blk_changed:
            blk.body = [x for x in new_body if x is not None]

    return func_ir


def _insert_build_map(
    func_ir, name, old_body, new_body, lit_old_idx, lit_new_idx, map_updates
):
    """
    Inserts a an assign with the given name into the new body using the
    information from dictionaries:
        lit_old_idx: name -> index in which the original build_map is found
        lit_new_idx: name -> index in which to insert
        map_updates: name -> key/value items for the new build map.

    After inserting into new_body, name is deleted from all of the dictionaries.
    """
    old_idx = lit_old_idx[name]
    new_idx = lit_new_idx[name]
    items = map_updates[name]
    # Insert each remaining dictionary to the earliest location it combined
    # its variables. This is to avoid error prone pattern matching in the IR,
    # especially with nodes expected to fall at the end of blocks.
    new_body[new_idx] = _build_new_build_map(func_ir, name, old_body, old_idx, items)
    del lit_old_idx[name]
    del lit_new_idx[name]
    del map_updates[name]


def _build_new_build_map(func_ir, name, old_body, old_lineno, new_items):
    """
    Create a new build_map with a new set of key/value items
    but all the other info the same.
    """
    old_assign = old_body[old_lineno]
    old_target = old_assign.target
    old_bm = old_assign.value
    # Build the literals
    literal_keys = []
    # Track the constant key/values to set the literal_value field of build_map properly
    values = []
    for pair in new_items:
        k, v = pair
        key_def = guard(get_definition, func_ir, k)
        if isinstance(key_def, (ir.Const, ir.Global, ir.FreeVar)):
            literal_keys.append(key_def.value)
        value_def = guard(get_definition, func_ir, v)
        if isinstance(value_def, (ir.Const, ir.Global, ir.FreeVar)):
            values.append(value_def.value)
        else:
            # Append unknown value if not a literal.
            values.append(numba.core.interpreter._UNKNOWN_VALUE(v.name))

    value_indexes = {}
    if len(literal_keys) == len(new_items):
        # All keys must be literals to have any literal values.
        literal_value = dict(zip(literal_keys, values))
        for i, k in enumerate(literal_keys):
            value_indexes[k] = i
    else:
        literal_value = None

    # Construct a new build map.
    new_bm = ir.Expr.build_map(
        items=new_items,
        size=len(new_items),
        literal_value=literal_value,
        value_indexes=value_indexes,
        loc=old_bm.loc,
    )

    func_ir._definitions[name].append(new_bm)

    # Return a new assign.
    return ir.Assign(new_bm, ir.Var(old_target.scope, name, old_target.loc), new_bm.loc)


def peep_hole_fix_build_list(func_ir):
    """Transform constant lists created via list append operations into a simple
    build_list expression"""
    for block in func_ir.blocks.values():
        _transform_list_appends(func_ir, block)


def _get_const_append_list_var_name(call_expr, func_ir, list_vars):
    """Match list.append(constant) pattern and return the list variable name and
    constant value.
    Return None if pattern doesn't match.
    """
    from numba.core.ir_utils import (
        GuardException,
        find_callname,
        guard,
    )

    from bodo.utils.transform import (
        get_const_value_inner,
    )

    fdef = guard(find_callname, func_ir, call_expr)
    if (
        fdef is None
        or fdef[0] != "append"
        or not isinstance(fdef[1], ir.Var)
        or (l_var := fdef[1].name) not in list_vars
    ):
        return None

    try:
        const = get_const_value_inner(func_ir, call_expr.args[0])
    except GuardException:
        return None

    return l_var, const


def _transform_list_appends(func_ir, block):
    """Transform constant lists created via list append operations into a simple
    build_list expression to facilitate pattern matching later.
    See read_csv() names argument of load_performance_data() in
    etl_data_conversion_mortgages example on Python 3.14.
    Example:
    l = []
    l.append(1)
    l.append(2)
    to:
    l = [1, 2]

    Args:
        func_ir: The function IR containing the block.
        block: The block of instructions to be transformed.
    """
    from numba.core.ir_utils import (
        mk_unique_var,
    )

    from bodo.utils.utils import (
        is_assign,
        is_call_assign,
        is_expr,
    )

    list_vars = {}
    new_body = []

    for inst in block.body:
        if (
            is_assign(inst)
            and is_expr(inst.value, "build_list")
            and len(inst.value.items) == 0
        ):
            list_vars[inst.target.name] = []
        elif (
            is_assign(inst)
            and is_expr(inst.value, "getattr")
            and inst.value.attr == "append"
        ):
            pass
        elif (
            is_call_assign(inst)
            and (
                append_info := _get_const_append_list_var_name(
                    inst.value, func_ir, list_vars
                )
            )
            is not None
        ):
            l_var, const = append_info
            list_vars[l_var].append(const)
        else:
            # Materialize any lists that are used in this instruction
            used_l_vars = list_vars.keys() & {v.name for v in inst.list_vars()}
            for l_var in used_l_vars:
                l_var_consts = list_vars.pop(l_var)
                # Avoid trivial or non-compilable cases
                if (
                    len(l_var_consts) == 0
                    or len({numba.typeof(c) for c in l_var_consts}) != 1
                ):
                    continue
                scope = inst.target.scope
                loc = inst.loc
                const_vars = []
                # Create variables for constants
                for c in l_var_consts:
                    const_assign = ir.Assign(
                        ir.Const(c, loc),
                        ir.Var(scope, mk_unique_var("append_const"), loc),
                        loc,
                    )
                    func_ir._definitions[const_assign.target.name].append(
                        const_assign.value
                    )
                    new_body.append(const_assign)
                    const_vars.append(const_assign.target)

                out_var = ir.Var(scope, l_var, loc)

                # Remove old const list append code
                updated_body = []
                list_append_func_vars = set()
                for i, stmt in enumerate(new_body):
                    if (
                        is_assign(stmt)
                        and is_expr(stmt.value, "build_list")
                        and len(stmt.value.items) == 0
                    ):
                        func_ir._definitions[l_var].remove(stmt.value)
                    elif (
                        is_assign(stmt)
                        and is_expr(stmt.value, "getattr")
                        and stmt.value.attr == "append"
                        and stmt.value.value.name == l_var
                    ):
                        list_append_func_vars.add(stmt.target.name)
                    elif (
                        is_call_assign(stmt)
                        and stmt.value.func.name in list_append_func_vars
                    ):
                        pass
                    else:
                        updated_body.append(stmt)

                new_body = updated_body

                new_assign = ir.Assign(
                    ir.Expr.build_list(const_vars, loc),
                    out_var,
                    loc,
                )
                func_ir._definitions[out_var.name].append(new_assign.value)
                new_body.append(new_assign)

        new_body.append(inst)

    block.body = new_body
