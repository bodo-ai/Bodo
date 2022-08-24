# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""There are a large number of operators that need a wrapper that returns null if any of the input arguments are null,
and otherwise return the result of the original function. This file automatically generates the tests for these library functions."""

import random


def generate_fn_test_strings(
    lib_fn_string,
    lib_fn_name,
    arg_values,
    expected_outputs,
    use_np_isclose,
    none_opt_testcase_index=None,
):
    """
    takes a string python function signature, a plaintext name for the function (specifically,
    a name that can used in a python function declaration, IE def test_{fnName}()),
    a nested list of function inputs in the form of arg_values[inputNo][argNo],
    a list of the expected outputs for each set of function inputs,
    in the form of expected_outputs[inputNo], and a boolean flag to determine if
    the function output and the expected output must be compared using np.isclose

    For example, assume we have
    arg_values = [[1,2,3], [3,2,1]]
    expected_outputs = [1, 3]
    we will generate tests assuming that fn(1,2,3) == 1 and fn(3,2,1) == 3
    (Using np.isclose instead of == if the flag is set)
    It is required that each nested list within arg_values has the same length

    For none/optional checks, we use one set of argument values. if none_opt_testcase_index is set,
    it will generate the none and optional tests using the inputs/outputs at that index.
    otherwise, the choice will be random

    This function generates the strings for a number of functions that test the default
    behavior, the None behavior, and the optional behavior of the JIT function.

    This is designed to be used in conjunction with generate_library_fn_string,
    but can be used for any function that has the same expected null behavior.
    """

    # performs some checks to insure that the passed in data is valid

    num_function_inputs = len(arg_values)
    assert num_function_inputs >= 1
    assert num_function_inputs == len(expected_outputs)
    num_args = len(arg_values[0])
    assert num_args >= 1

    # each sub list should have the same length
    for i in range(1, num_function_inputs):
        assert num_args == len(arg_values[i])

    default_tests = generate_default_tests(
        lib_fn_string, lib_fn_name, arg_values, expected_outputs, use_np_isclose
    )

    # For none/optional checks, we use one set of argument values
    # Since the choice is arbitrary, we select a random value to use for
    # testing. In the event that a randomly selected input set causes an
    # error on nightly, we can always check every possible input by
    # seting the optional argument manually
    if none_opt_testcase_index == None:
        none_opt_testcase_index = random.randint(0, num_function_inputs - 1)

    none_opt_testcase_input = arg_values[none_opt_testcase_index]
    none_opt_testcase_expeceted_output = expected_outputs[none_opt_testcase_index]

    none_tests = generate_none_tests(
        lib_fn_string, lib_fn_name, none_opt_testcase_input
    )

    opt_tests = generate_optional_tests(
        lib_fn_string,
        lib_fn_name,
        none_opt_testcase_input,
        none_opt_testcase_expeceted_output,
        use_np_isclose,
    )

    return (default_tests, none_tests, opt_tests)


def generate_default_tests(
    lib_fn_string, lib_fn_name, arg_values, expected_outputs, use_np_isclose
):
    """
    takes a string python function, a plaintext name for the function (specifically,
    a name that can used in a python function declaration, IE def test_{fnName}()),
    a nested list of function inputs in the form of arg_values[inputNo][argNo],
    a list of the expected outputs for each set of function inputs,
    in the form of expected_outputs[inputNo], and a boolean flag to determine if
    the function output and the expected output must be compared using np.isclose

    For example, assume we have
    arg_values = [[1,2,3], [3,2,1]]
    expected_outputs = [1, 3]
    we will generate tests assuming that fn(1,2,3) == 1 and fn(3,2,1) == 3
    (Using np.isclose instead of == if the flag is set)
    It is required that each nested list within arg_values has the same length

    Returns a list of string functions that check that each set of input argument values
    yields the expected output for the specified function.
    """
    num_provided_function_inputs = len(arg_values)
    num_args = len(arg_values[0])
    tests = []
    for cur_input_index in range(0, num_provided_function_inputs):
        cur_inputs = []
        cur_output = expected_outputs[cur_input_index]
        for cur_arg_index in range(0, num_args):
            cur_inputs.append(arg_values[cur_input_index][cur_arg_index])
        cur_inputs_string = ", ".join(cur_inputs)
        if use_np_isclose:
            tests.append(
                f"""
def test_{lib_fn_name}_default_input_{cur_input_index}():
    assert np.isclose({lib_fn_string}({cur_inputs_string}), {cur_output})\n"""
            )
        else:
            tests.append(
                f"""
def test_{lib_fn_name}_default_input_{cur_input_index}():
    assert {lib_fn_string}({cur_inputs_string}) == {cur_output}\n"""
            )

    return tests


def generate_none_tests(lib_fn_string, lib_fn_name, arg_values):
    """
    Takes a string python function, a plaintext name for the function (specifically,
    a name that can used in a python function declaration, IE def test_{fnName}()),
    and a list of function inputs in the form of arg_values[argNo]

    for example, if arg_values = [1, 2, 3], fn(1,2,3) should be valid

    Returns the a list of testing functions strings that will check the behavior of the
    specified function on None values.
    """

    num_args = len(arg_values)

    # generates tests that check what happens when each argument is none.
    # this can be exteneded in the future to check every permutation of None inputs if needed
    none_tests = []
    for none_arg_index in range(0, num_args):
        cur_inputs = []
        for arg_index in range(0, num_args):
            if arg_index == none_arg_index:
                cur_inputs.append("None")
            else:
                cur_inputs.append(arg_values[arg_index])
        cur_inputs_string = ", ".join(cur_inputs)
        none_tests.append(
            f"""
def test_{lib_fn_name}_None_Arg_{none_arg_index}():
    assert {lib_fn_string}({cur_inputs_string}) is None\n"""
        )

    return none_tests


def generate_optional_tests(
    lib_fn_string, lib_fn_name, arg_values, expected_output, use_np_isclose
):
    """
    Takes a string python function, a plaintext name for the function (specifically,
    a name that can used in a python function declaration, IE def test_{fnName}()),
    a list of function inputs in the form of arg_values[argNo],the
    expected output of the function for those inputs, and a boolean flag to determine if
    the function output and the expected output must be compared using np.isclose

    for example, if arg_values = [1, 2, 3], fn(1,2,3) == expected_output
    (Using np.isclose instead of == if the flag is set)

    Returns a list of testing function strings that will check the behavior of the
    specified function on optional types.

    """
    indent = "    "
    num_args = len(arg_values)

    # currently, generatets tests that check what occurs if each input is optional,
    # and if all inputs are optional. can be extended in the future if need be
    internal_fn_arg_names = [f"arg{argno}" for argno in range(num_args)]
    internal_fn_args_list = ", ".join(internal_fn_arg_names)

    set_allargs_none_string = f"\n{indent*4}".join(
        [f"{internal_fn_arg_names[argno]} = None" for argno in range(num_args)]
    )

    optional_generator_fn_defn = f"""
    @bodo.jit
    def {lib_fn_name}_run_with_optional_args(flag, {internal_fn_args_list}, optional_num):
            """

    # generate the if/else code that sets each argument to optional
    # depending on the value of optional_num
    for argno in range(num_args + 1):
        if argno == 0:
            optional_case = f"""
        if optional_num == {argno}:
            if flag:
                {internal_fn_arg_names[argno]} = None
            return {lib_fn_string}({internal_fn_args_list})
    """

        elif argno == num_args:
            optional_case = f"""
        else:
            if flag:
                {set_allargs_none_string}
            return {lib_fn_string}({internal_fn_args_list})
    """

        else:
            optional_case = f"""
        elif optional_num == {argno}:
            if flag:
                {internal_fn_arg_names[argno]} = None
            return {lib_fn_string}({internal_fn_args_list})
    """

        optional_generator_fn_defn = optional_generator_fn_defn + optional_case

    fn_inputs_string = ", ".join(arg_values)
    test_strings = []
    for argno in range(num_args + 1):
        if use_np_isclose:
            test_strings.append(
                f"""
def test_{lib_fn_name}_optional_num_{argno}():
{optional_generator_fn_defn}
    assert np.isclose({lib_fn_name}_run_with_optional_args(False, {fn_inputs_string}, {argno}), {expected_output})
    assert {lib_fn_name}_run_with_optional_args(True, {fn_inputs_string}, {argno}) is None\n"""
            )
        else:
            test_strings.append(
                f"""
def test_{lib_fn_name}_optional_num_{argno}():
{optional_generator_fn_defn}
    assert {lib_fn_name}_run_with_optional_args(False, {fn_inputs_string}, {argno}) == {expected_output}
    assert {lib_fn_name}_run_with_optional_args(True, {fn_inputs_string}, {argno}) is None\n"""
            )

    return [optional_generator_fn_defn] + test_strings
