# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" There are a large number of operators that need a wrapper that returns null if any of the input arguments are null,
and otherwise return the result of the original function. This file automatically generates these library functions."""
import operator

from numba import generated_jit

import bodo


def generate_library_fn_string(fn_name, fn_expr_lambda, num_args):
    """
    fn_name must be a valid plaintext string that can be used in the declaration of the new library function,
    IE def {fn_name}(...) must be valid python. fn_expr is lambda function that will return a string function call when passed
    a list of arguments. For example, the fn_expr_lambda for ltrim when passed the argument ["' hello '"] should return the
    string "' hello '.ltrim". the fn_expr_lambda for upper when passed the same argument, should return the string
    "upper(' hello ')" Num_args is the number of arguments of the original function

    This function takes the specified python function, and generates a library JIT function wrapper that performs null
    checking, and returns None if any of the args are found to be null. For example,

    generate_library_fn_string('custom_fn', lambda x: f'mymodule.custom_fn{", ".join(x)}', 3) should yield the following string:

    @generated_jit(nopython=True)
    def __bodosql__custom_fn(arg0, arg1, arg2):
        \"automatically generated library function for custom\"

        #if either input is None, return None
        if (arg0 == bodo.none or arg1 == bodo.none or arg2 == bodo.none):
            return lambda arg0, arg1, arg2: None

        # If either input is optional, the output is optional.
        # We could merge this code path with the default, but
        # if we can avoid optional types we should.
        elif ({any_arg_is_optional_condition}):
            def impl(arg0, arg1, arg2):
                if (arg0 is None or arg1 is None or arg2 is None):
                    return None
                else:
                    # Call internal bodo function that changes the converts the
                    # type of Optional(type) to just type. If a or b isn't optional
                    # this is basically a noop
                    arg0 = bodo.utils.indexing.unoptional(arg0)
                    arg1 = bodo.utils.indexing.unoptional(arg1)
                    arg2 = bodo.utils.indexing.unoptional(arg2)
                    return mymodule.custom_fn(arg0, arg1, arg2)
            return impl
        else:
            return lambda arg0, arg1, arg2: mymodule.custom_fn(arg0, arg1, arg2)
    """

    args_strings = [f"arg{x}" for x in range(0, num_args)]
    fn_expr = fn_expr_lambda(args_strings)
    args_list = ", ".join(args_strings)

    any_arg_bodo_none_condition = " or ".join(
        [f"{cur_arg} == bodo.none" for cur_arg in args_strings]
    )
    any_arg_is_optional_condition = " or ".join(
        [f"isinstance({cur_arg}, bodo.optional)" for cur_arg in args_strings]
    )
    any_arg_is_none_condition = " or ".join(
        [f"{cur_arg} is None" for cur_arg in args_strings]
    )

    indent = "    "
    convert_optional_args_code = "\n".join(
        [
            f"{indent * 4}{cur_arg} = bodo.utils.indexing.unoptional({cur_arg})"
            for cur_arg in args_strings
        ]
    )
    lib_fn_name = bodosql_library_fn_name(fn_name)

    function_code = f"""
@generated_jit(nopython=True)
def {lib_fn_name}({args_list}):
    \"automatically generated library function for {fn_name}\"

    #if either input is None, return None
    if ({any_arg_bodo_none_condition}):
        return lambda {args_list}: None

    # If either input is optional, the output is optional.
    # We could merge this code path with the default, but
    # if we can avoid optional types we should.
    elif ({any_arg_is_optional_condition}):
        def impl({args_list}):
            if ({any_arg_is_none_condition}):
                return None
            else:
                # Call internal bodo function that changes the converts the
                # type of Optional(type) to just type. If a or b isn't optional
                # this is basically a noop
{convert_optional_args_code}
                return {fn_expr}
        return impl

    else:
        return lambda {args_list}: {fn_expr}
        """
    return function_code


def generate_fn_impl(fn_name, fn_expr_lambda, num_args):
    "returns a library function that wrapps the specified function"
    func_text = generate_library_fn_string(fn_name, fn_expr_lambda, num_args)
    lib_fn_name = bodosql_library_fn_name(fn_name)
    locs = {}
    exec(
        func_text,
        {"bodo": bodo, "operator": operator, "generated_jit": generated_jit},
        locs,
    )
    func = locs[lib_fn_name]
    return func


def bodosql_library_fn_name(fn_name):
    """Returns the name of the generated library function"""
    return f"sql_null_checking_{fn_name}"
