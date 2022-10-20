# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements BodoSQL array kernels related to JSON utilities
"""

import numba
from numba.core import types

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


@numba.generated_jit(nopython=True)
def parse_json(arg):
    """Handles cases where parse_json receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    if isinstance(arg, types.optional):  # pragma: no cover
        return bodo.libs.bodosql_array_kernel_utils.unopt_argument(
            "bodo.libs.bodosql_array_kernels.parse_json", ["arg"], 0
        )

    def impl(arg):  # pragma: no cover
        return parse_json_util(arg)

    return impl


@numba.generated_jit(nopython=True)
def parse_single_json_map(s):
    """Takes in a scalar string and parses it to a JSON dictionary, returning
    the dictionary (or None if malformed.)

    Implemented by a 9-state automata with the states described here:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1162772485/Handling+JSON+for+BodoSQL
    """

    def impl(s):  # pragma: no cover
        state = 1
        result = {}
        stack = ["{"]
        current_key = ""
        current_value = ""
        escaped = False
        for char in s:

            # START
            if state == 1:
                if char.isspace():
                    continue
                elif char == "{":
                    state = 2
                else:
                    return None

            # KEY_HUNT
            elif state == 2:
                if char.isspace():
                    continue
                elif char == '"':
                    state = 3
                elif char == "}":
                    state = 9
                else:
                    return None

            # KEY_FILL
            elif state == 3:
                if escaped:
                    current_key += char
                    escaped = False
                elif char == '"':
                    state = 4
                elif char == "\\":
                    escaped = True
                else:
                    current_key += char

            # COLON_HUNT
            elif state == 4:
                if char.isspace():
                    continue
                elif char == ":":
                    state = 5
                else:
                    return None

            # VALUE_HUNT
            elif state == 5:
                if char.isspace():
                    continue
                if char in "},]":
                    return None
                else:
                    state = 7 if char == '"' else 6
                    current_value += char
                    if char in "{[":
                        stack.append(char)

            # VALUE_FILL
            elif state == 6:
                if char.isspace():
                    continue
                if char in "{[":
                    current_value += char
                    stack.append(char)
                elif char in "}]":
                    revChar = "{" if char == "}" else "["
                    if len(stack) == 0 or stack[-1] != revChar:
                        return None
                    elif len(stack) == 1:
                        result[current_key] = current_value
                        current_key = ""
                        current_value = ""
                        stack.pop()
                        state = 9
                    elif len(stack) == 2:
                        current_value += char
                        result[current_key] = current_value
                        current_key = ""
                        current_value = ""
                        stack.pop()
                        state = 8
                    else:
                        current_value += char
                        stack.pop()
                elif char == '"':
                    current_value += char
                    state = 7
                elif char == ",":
                    if len(stack) == 1:
                        result[current_key] = current_value
                        current_key = ""
                        current_value = ""
                        state = 2
                    else:
                        current_value += char
                else:
                    current_value += char

            # STRING_FILL
            elif state == 7:
                if escaped:
                    current_value += char
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    current_value += char
                    state = 6
                else:
                    current_value += char

            # COMMA_HUNT
            elif state == 8:
                if char.isspace():
                    continue
                elif char == ",":
                    state = 2
                elif char == "}":
                    state = 9
                else:
                    return None

            # FINISH
            elif state == 9:
                if not char.isspace():
                    return None

        # Output the map so long as we ended up in the FINISH phase
        return result if state == 9 else None

    return impl


@numba.generated_jit(nopython=True)
def parse_json_util(arr):
    """A dedicated kernel for the SQL function PARSE_JSON which takes in a string
       and returns the string converted to JSON objects.

       Currently always converts to a map of type str:str. All values are stored
       as strings to be parsed/casted at a later time.


    Args:
        arr (string scalar/array): the string(s) that is being converted to JSON

    Returns:
        map[str, str] (or array of map[str, str]): the string(s) converted to JSON
    """

    bodo.libs.bodosql_array_kernels.verify_string_arg(arr, "PARSE_JSON", "s")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [False]

    scalar_text = "jmap = bodo.libs.bodosql_json_array_kernels.parse_single_json_map(arg0) if arg0 is not None else None\n"
    if bodo.utils.utils.is_array_typ(arr, True):
        prefix_code = "lengths = bodo.utils.utils.alloc_type(n, bodo.int32, (-1,))\n"
        scalar_text += "res.append(jmap)\n"
        scalar_text += "if jmap is None:\n"
        scalar_text += "   lengths[i] = 0\n"
        scalar_text += "else:\n"
        scalar_text += "   lengths[i] = len(jmap)\n"
    else:
        prefix_code = None
        scalar_text += "return jmap"

    suffix_code = (
        "res2 = bodo.libs.map_arr_ext.pre_alloc_map_array(n, lengths, out_dtype)\n"
    )
    suffix_code += "numba.parfors.parfor.init_prange()\n"
    suffix_code += "for i in numba.parfors.parfor.internal_prange(n):\n"
    suffix_code += "   if res[i] is None:\n"
    suffix_code += "     bodo.libs.array_kernels.setna(res2, i)\n"
    suffix_code += "   else:\n"
    suffix_code += "     res2[i] = res[i]\n"
    suffix_code += "res = res2\n"

    struct_type = bodo.StructArrayType(
        (bodo.string_array_type, bodo.string_array_type), ("key", "value")
    )
    out_dtype = bodo.utils.typing.to_nullable_type(struct_type)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
        suffix_code=suffix_code,
        res_list=True,
        support_dict_encoding=False,
    )
