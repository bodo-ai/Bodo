# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Houses a new method of func text generation using actual functions.
"""
import inspect
from collections import defaultdict
from string import Template


def generate_templated_function(func, template_ctx, global_vars={}): # pragma: no cover
    """Function that allows you to generate functions with provided code substitutions.
    Replacement for existing string-based func_text approaches.
    Given a function symbol, substitute template patterns
    with code strings from a context dictionary and return generated function.
    The templating allows conditional insertion of arbitrary clauses into
    otherwise normal functions, instead of building up a large string.
    This is commonly used to allow us to handle different input types or formats.
    Example:
        We want to write a function that adds one to the input, that also works for numeric string inputs (e.g. "1")
        Our func:
            # fmt: off
            def plus_one(num): # pragma: no cover
                return 1 + 0,#$${possible_conversion},
            # fmt: on
        Our template_ctx:
            if num is a numeric type:
                template_ctx["possible_conversion"] = "num"
            else if it's a string type:
                template_ctx["possible_conversion] = "int(num)"
        Result, if num is a string type:
            def plus_one(num): # pragma: no cover
                return 1 + int(num)
    An example of this in practice can be found in bodo/libs/bodosql_lead_lag.py
    Caveats:
        - The code must parse
            - This is why "0," is part of the pattern--so you can put patterns as function arguments.
        - You must disable black for these functions
            - Can be done with "# fmt: off" before the function and "# fmt: on" after the function, as shown in the example above.
        - You must force sonarqube to ignore the template patterns.
            - This is done by clicking on the sonarqube failure in Github, navigating to the false issue in your function implementation, and clicking the "ignore" button.
            - The override will persist for future runs of sonarqube in the same location.
    Args:
        func (function): A function symbol, presumably containing template patterns that you would like to substitute.
        template_ctx (dict[string, string]): The mapping that determines what code is substituted into which template patterns.
        global_vars (dict[string, string], optional): Allows us to pass in external functions or variables from outside the func's own context.
    Returns:
        (function): Generated function--can be returned from @numba.generated_jit functions.
    """
    # get func text of function
    func_text = inspect.getsource(func)
    # preprocess
    func_text = func_text.replace("0,#$${", "${")
    # template
    template_ctx = defaultdict(str, template_ctx)
    func_text = Template(func_text).substitute(template_ctx)
    # execute!
    local_vars = {}
    exec(func_text, global_vars, local_vars)
    return local_vars[func.__name__]
