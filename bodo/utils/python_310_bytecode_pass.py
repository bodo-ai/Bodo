# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
transforms the IR to handle bytecode issues in Python 3.10. This
should be removed once https://github.com/numba/numba/pull/7866
is included in Numba 0.56
"""
import operator

import numba
from numba.core import ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.errors import UnsupportedError
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
        peep_hole_call_function_ex_to_call_function_kw(state.func_ir)
        peep_hole_fuse_dict_add_updates(state.func_ir)
        peep_hole_fuse_tuple_adds(state.func_ir)
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


# The code below is all copied from https://github.com/numba/numba/pull/7866


def _call_function_ex_replace_kws_small(keyword_expr, new_body, buildmap_idx):
    """
    Extracts the kws args passed as varkwarg
    for CALL_FUNCTION_EX. This pass is taken when
    n_kws <= 15 and the bytecode looks like:

        # Start for each argument
        LOAD_FAST  # Load each argument.
        # End for each argument
        ...
        BUILD_CONST_KEY_MAP # Build a map

    In the IR generated, the varkwarg refer
    to a single build_map that contains all of the
    kws. In addition to returning the kws, this
    function updates new_body to remove all usage
    of the map.
    """
    kws = keyword_expr.items.copy()
    # kws are required to have constant keys.
    # We update these with the value_indexes
    value_indexes = keyword_expr.value_indexes
    for key, index in value_indexes.items():
        kws[index] = (key, kws[index][1])
    # Remove the build_map by setting the list
    # index to None. Nones will be removed later.
    new_body[buildmap_idx] = None
    return kws


def _call_function_ex_replace_kws_large(
    body, buildmap_name, buildmap_idx, search_end, new_body
):
    """
    Extracts the kws args passed as varkwarg
    for CALL_FUNCTION_EX. This pass is taken when
    n_kws <= 15 and the bytecode looks like:

        BUILD_MAP # Construct the map
        # Start for each argument
        LOAD_CONST # Load a constant for the name of the argument
        LOAD_FAST  # Load each argument.
        MAP_ADD # Append the (key, value) pair to the map
        # End for each argument

    In the IR generated, the initial build map is empty and a series
    of setitems are applied afterwards. THE IR looks like:

    Finally at the end the bytecode will contain a CALL_FUNCTION_EX instruction.

        $constvar = const(str, ...) # create the const key
        # CREATE THE ARGUMENT, This may take multiple lines.
        $var = getattr(
            value=$build_map_var,
            attr=__setitem__,
        )
        $unused_var = call $var($constvar, $created_arg)

    We iterate through the IR, deleting all usages of the buildmap
    from the new_body, and adds the kws to a new kws list.
    """
    errmsg = "CALL_FUNCTION_EX with **kwargs not supported"
    # Remove the build_map from the body.
    new_body[buildmap_idx] = None
    kws = []
    search_start = buildmap_idx + 1
    while search_start <= search_end:
        # The first value must be a constant.
        const_stmt = body[search_start]
        if not (
            isinstance(const_stmt, ir.Assign) and isinstance(const_stmt.value, ir.Const)
        ):
            # We cannot handle this format so raise the
            # original error message.
            raise UnsupportedError(errmsg)
        key_var_name = const_stmt.target.name
        key_val = const_stmt.value.value
        search_start += 1
        # Now we need to search for a getattr with setitem
        not_found_getattr = True
        while search_start <= search_end and not_found_getattr:
            getattr_stmt = body[search_start]
            if (
                isinstance(getattr_stmt, ir.Assign)
                and isinstance(getattr_stmt.value, ir.Expr)
                and getattr_stmt.value.op == "getattr"
                and (getattr_stmt.value.value.name == buildmap_name)
                and getattr_stmt.value.attr == "__setitem__"
            ):
                not_found_getattr = False
            else:
                search_start += 1
        if not_found_getattr or search_start == search_end:
            # We cannot handle this format so raise the
            # original error message.
            raise UnsupportedError(errmsg)
        setitem_stmt = body[search_start + 1]
        if not (
            isinstance(setitem_stmt, ir.Assign)
            and isinstance(setitem_stmt.value, ir.Expr)
            and setitem_stmt.value.op == "call"
            and (setitem_stmt.value.func.name == getattr_stmt.target.name)
            and len(setitem_stmt.value.args) == 2
            and (setitem_stmt.value.args[0].name == key_var_name)
        ):
            # We cannot handle this format so raise the
            # original error message.
            raise UnsupportedError(errmsg)
        arg_var = setitem_stmt.value.args[1]
        # Append the (key, value) pair.
        kws.append((key_val, arg_var))
        # Remove the __setitem__ getattr and call
        new_body[search_start] = None
        new_body[search_start + 1] = None
        search_start += 2
    return kws


def _call_function_ex_replace_args_small(tuple_expr, new_body, buildtuple_idx):
    """
    Extracts the args passed as vararg
    for CALL_FUNCTION_EX. This pass is taken when
    n_args <= 30 and the bytecode looks like:

        # Start for each argument
        LOAD_FAST  # Load each argument.
        # End for each argument
        ...
        BUILD_TUPLE # Create a tuple of the arguments

    In the IR generated, the vararg refer
    to a single build_tuple that contains all of the
    args. In addition to returning the args, this
    function updates new_body to remove all usage
    of the tuple.
    """
    # Delete the build tuple
    new_body[buildtuple_idx] = None
    # Return the args.
    return tuple_expr.items


def _call_function_ex_replace_args_large(vararg_stmt, body, new_body, search_end):
    """
    Extracts the args passed as vararg
    for CALL_FUNCTION_EX. This pass is taken when
    n_args > 30 and the bytecode looks like:

        BUILD_TUPLE # Create a list to append to
        # Start for each argument
        LOAD_FAST  # Load each argument.
        LIST_APPEND # Add the argument to the list
        # End for each argument
        ...
        LIST_TO_TUPLE # Convert the args to a tuple.

    In the IR generated, the tuple is created by concatenating
    together several 1 elem tuples to an initial empty tuple.
    We traverse backwards in the IR, collecting args, until we
    find the original empty tuple. For example, the IR might
    look like:

        $orig_tuple = build_tuple(items=[])
        $first_var = build_tuple(items=[Var(arg0, test.py:6)])
        $next_tuple = $orig_tuple + $first_var
        ...
        $final_var = build_tuple(items=[Var(argn, test.py:6)])
        $final_tuple = $prev_tuple + $final_var
        $varargs_var = $final_tuple

    It unclear if the extra assignment at the end is always present.
    In addition to collecting and returning the original args, we also
    delete any IR statments that uses any of the tuples from new_body.
    """
    errmsg = "CALL_FUNCTION_EX with **kwargs not supported"
    # We traverse to the front of the block to look for the original
    # tuple.
    search_start = 0
    total_args = []
    if isinstance(vararg_stmt, ir.Assign) and isinstance(vararg_stmt.value, ir.Var):
        target_name = vararg_stmt.value.name
        # If there is an initial assignment, delete it
        new_body[search_end] = None
        search_end -= 1
    else:
        target_name = vararg_stmt.target.name
    start_not_found = True
    # Traverse backwards to find all concatentations
    # until eventually reaching the original empty tuple.
    while search_end >= search_start and start_not_found:
        concat_stmt = body[search_end]
        if (
            isinstance(concat_stmt, ir.Assign)
            and concat_stmt.target.name == target_name
            and isinstance(concat_stmt.value, ir.Expr)
            and concat_stmt.value.op == "build_tuple"
            and not concat_stmt.value.items
        ):
            # If we have reached the build_tuple
            # we exit.
            start_not_found = False
            new_body[search_end] = None
        else:
            # We expect to find another arg to append.
            # The first stmt must be an add
            if (search_end == search_start) or not (
                isinstance(concat_stmt, ir.Assign)
                and (concat_stmt.target.name == target_name)
                and isinstance(concat_stmt.value, ir.Expr)
                and concat_stmt.value.op == "binop"
                and concat_stmt.value.fn == operator.add
            ):
                # We cannot handle this format.
                raise UnsupportedError(errmsg)
            lhs_name = concat_stmt.value.lhs.name
            rhs_name = concat_stmt.value.rhs.name
            # The previous statment should be a
            # build_tuple containing the arg.
            arg_tuple_stmt = body[search_end - 1]
            if not (
                isinstance(arg_tuple_stmt, ir.Assign)
                and isinstance(arg_tuple_stmt.value, ir.Expr)
                and (arg_tuple_stmt.value.op == "build_tuple")
                and len(arg_tuple_stmt.value.items) == 1
            ):
                # We cannot handle this format.
                raise UnsupportedError(errmsg)
            if arg_tuple_stmt.target.name == lhs_name:
                target_name = rhs_name
            elif arg_tuple_stmt.target.name == rhs_name:
                target_name = lhs_name
            else:
                # We cannot handle this format.
                raise UnsupportedError(errmsg)
            total_args.append(arg_tuple_stmt.value.items[0])
            new_body[search_end] = None
            new_body[search_end - 1] = None
            search_end -= 2
            # Avoid any space between appends
            keep_looking = True
            while search_end >= search_start and keep_looking:
                next_stmt = body[search_end]
                if isinstance(next_stmt, ir.Assign) and (
                    next_stmt.target.name == target_name
                ):
                    keep_looking = False
                else:
                    search_end -= 1
    if start_not_found:
        # We cannot handle this format so raise the
        # original error message.
        raise UnsupportedError(errmsg)
    # Reverse the arguments so we get the correct order.
    return total_args[::-1]


def peep_hole_call_function_ex_to_call_function_kw(func_ir):
    """
    This peephole rewrites a bytecode sequence unique to Python 3.10
    where CALL_FUNCTION_EX is used instead of CALL_FUNCTION_KW because of
    stack limitations set by CPython. This limitation is imposed whenever
    a function call has too many arguments or keyword arguments.

    https://github.com/python/cpython/blob/a58ebcc701dd6c43630df941481475ff0f615a81/Python/compile.c#L55
    https://github.com/python/cpython/blob/a58ebcc701dd6c43630df941481475ff0f615a81/Python/compile.c#L4442

    In particular, change is imposed whenever (n_args / 2) + n_kws > 15.

    Different bytecode is generated for args depending on if n_args > 30
    or n_args <= 30 and similarly if n_kws > 15 or n_kws <= 15.

    This function unwraps the *args and **kwargs in the function call
    and places these values directly into the args and kwargs of the call.
    """
    # All changes are local to the a single block
    # so it can be traversed in any order.
    errmsg = "CALL_FUNCTION_EX with **kwargs not supported"
    for blk in func_ir.blocks.values():
        blk_changed = False
        new_body = []
        for i, stmt in enumerate(blk.body):
            if (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Expr)
                and stmt.value.op == "call"
                and stmt.value.varkwarg is not None
            ):
                blk_changed = True
                call = stmt.value
                args = call.args
                kws = call.kws
                # We search for replace if a call has either vararg
                # or varkwarg.
                vararg = call.vararg
                varkwarg = call.varkwarg
                start_search = i - 1
                # varkwarg should be defined second so we start there.
                varkwarg_loc = start_search
                keyword_def = None
                not_found = True
                while varkwarg_loc >= 0 and not_found:
                    keyword_def = blk.body[varkwarg_loc]
                    if (
                        isinstance(keyword_def, ir.Assign)
                        and keyword_def.target.name == varkwarg.name
                    ):
                        not_found = False
                    else:
                        varkwarg_loc -= 1
                if (
                    kws
                    or not_found
                    or not (
                        isinstance(keyword_def.value, ir.Expr)
                        and keyword_def.value.op == "build_map"
                    )
                ):
                    # If we couldn't find where the kwargs are created
                    # then it should be a normal **kwargs call
                    # so we produce an unsupported message.
                    raise UnsupportedError(errmsg)
                # Determine the kws
                if keyword_def.value.items:
                    # n_kws <= 15 case.
                    # Here the IR looks like a series of
                    # constants, then the arguments and finally
                    # a build_map that contains all of the pairs.
                    # For Example:
                    #
                    #   $const_n = const("arg_name")
                    #   $arg_n = ...
                    #   $kwargs_var = build_map(items=[
                    #              ($const_0, $arg_0),
                    #              ...,
                    #              ($const_n, $arg_n),])
                    kws = _call_function_ex_replace_kws_small(
                        keyword_def.value,
                        new_body,
                        varkwarg_loc,
                    )
                else:
                    # n_kws > 15 case.
                    # Here the IR is an initial empty build_map
                    # followed by a series of setitems with a constant
                    # key and then the argument.
                    # For example:
                    #
                    #   $kwargs_var = build_map(items=[])
                    #   $const_0 = const("arg_name")
                    #   $arg_0 = ...
                    #   $my_attr = getattr(const_0, attr=__setitem__)
                    #   $unused_var = call $my_attr($const_0, $arg_0)
                    #   ...
                    kws = _call_function_ex_replace_kws_large(
                        blk.body,
                        varkwarg.name,
                        varkwarg_loc,
                        i - 1,
                        new_body,
                    )
                start_search = varkwarg_loc
                # Vararg isn't required to be provided.
                if vararg is not None:
                    if args:
                        # If we have vararg then args is expected to
                        # be an empty list.
                        raise UnsupportedError(errmsg)
                    vararg_loc = start_search
                    args_def = None
                    not_found = True
                    while vararg_loc >= 0 and not_found:
                        args_def = blk.body[vararg_loc]
                        if (
                            isinstance(args_def, ir.Assign)
                            and args_def.target.name == vararg.name
                        ):
                            not_found = False
                        else:
                            vararg_loc -= 1
                    if not_found:
                        # If we couldn't find where the args are created
                        # then we can't handle this format.
                        raise UnsupportedError(errmsg)
                    if (
                        isinstance(args_def.value, ir.Expr)
                        and args_def.value.op == "build_tuple"
                    ):
                        # n_args <= 30 case.
                        # Here the IR is a simple build_tuple containing
                        # all of the args.
                        # For example:
                        #
                        #  $arg_n = ...
                        #  $varargs = build_tuple(
                        #   items=[$arg_0, ..., $arg_n]
                        #  )
                        args = _call_function_ex_replace_args_small(
                            args_def.value, new_body, vararg_loc
                        )
                    else:
                        # Here the IR is an initial empty build_tuple.
                        # Then for each arg, a new tuple with a single
                        # element is created and one by one these are
                        # added to a growing tuple.
                        # For example:
                        #
                        #  $combo_tup_0 = build_tuple(items=[])
                        #  $arg0 = ...
                        #  $arg0_tup = build_tuple(items=[$arg0])
                        #  $combo_tup_1 = $combo_tup_0 + $arg0_tup
                        #  $arg1 = ...
                        #  $arg1_tup = build_tuple(items=[$arg1])
                        #  $combo_tup_2 = $combo_tup_1 + $arg1_tup
                        #  ...
                        #  $combo_tup_n = $combo_tup_{n-1} + $argn_tup
                        #
                        # In addition, the IR seems to contain a final
                        # assignment for the varargs that looks like:
                        #
                        #  $varargs_var = $combo_tup_n
                        #
                        # Here args_def is expected to be a simple assignment.
                        # However, it is unclear if this extra assignment is
                        # always generated, so as a result the code is written
                        # to support args_def being either $varargs_var
                        # or $combo_tup_n from the above example.
                        args = _call_function_ex_replace_args_large(
                            args_def, blk.body, new_body, vararg_loc
                        )
                # Create a new call updating the args and kws
                new_call = ir.Expr.call(
                    call.func, args, kws, call.loc, target=call.target
                )
                if (
                    stmt.target.name in func_ir._definitions
                    and len(func_ir._definitions[stmt.target.name]) == 1
                ):
                    # if there's a single definition, drop it
                    func_ir._definitions[stmt.target.name].clear()
                func_ir._definitions[stmt.target.name].append(new_call)
                # Update the statement
                stmt = ir.Assign(new_call, stmt.target, stmt.loc)

            new_body.append(stmt)
        # Update the block body if we updated the IR
        if blk_changed:
            blk.body = [x for x in new_body if x is not None]
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
    that combines the two dicitonaries.
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
        literal_value = {x: y for x, y in zip(literal_keys, values)}
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
