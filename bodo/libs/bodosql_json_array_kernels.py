# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements BodoSQL array kernels related to JSON utilities
"""

from typing import List, Tuple, Union

import numba
from numba.core import types
from numba.extending import register_jitable

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_str,
    raise_bodo_error,
    to_str_arr_if_dict_array,
)


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


@numba.generated_jit(nopython=True)
def json_extract_path_text(data, path):
    """Handles cases where JSON_EXTRACT_PATH_TEXT receives optional arguments and
    forwards to args appropriate version of the real implementation"""
    args = [data, path]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return bodo.libs.bodosql_array_kernel_utils.unopt_argument(
                "bodo.libs.bodosql_array_kernels.json_extract_path_text",
                ["data", "path"],
                i,
            )

    def impl(data, path):  # pragma: no cover
        return json_extract_path_text_util(data, path)

    return impl


@numba.generated_jit(nopython=True)
def json_extract_path_text_util(data, path):
    """A dedicated kernel for the SQL function JSON_EXTRACT_PATH_TEXT which takes
       in a string representing JSON data and another string representing a
       sequence of extraction commands and uses them to extract a string value
       from the first string. Returns NULL if the path specified is not found.

    Args:
        data (string scalar/array): the string(s) representing JSON data
        path (string scalar/array): the string(s) representing the path from
        the JSON data that is to be extracted

    Returns:
        string scalar/array: the string(s) extracted from data

    Some examples:

    data = '''
        {
            "Name": "Daemon Targaryen",

            "Location": {

                "Continent": "Westeros",

                "Castle": "Dragonstone"

            },

            "Spouses": [

                {"First": "Rhea", "Last": "Royce"},

                {"First": "Laena", "Last": "Velaryon"},

                {"First": "Rhaenyra", "Last": "Targaryen"}

            ],

            "Age": 40

        }'''

    If path = 'Name', returns "Daemon Targaryen"

    If path = 'Location.Continent', returns "Westeros"

    If path = 'Spouses[2].First', returns "Rhaenyra"

    If path = 'Birthday', returns None (since that path does not exist)

    The specification should obey this document: https://docs.snowflake.com/en/sql-reference/functions/json_extract_path_text.html

    The details are described in this document: https://bodo.atlassian.net/wiki/spaces/B/pages/1162772485/Handling+JSON+for+BodoSQL
    """

    verify_string_arg(data, "JSON_EXTRACT_PATH_TEXT", "data")
    verify_string_arg(path, "JSON_EXTRACT_PATH_TEXT", "path")

    arg_names = ["data", "path"]
    arg_types = [data, path]
    propagate_null = [True] * 2
    scalar_text = "result = bodo.libs.bodosql_array_kernels.parse_and_extract_json_string(arg0, arg1)\n"
    scalar_text += "if result is None:\n"
    scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = result"
    out_dtype = bodo.string_array_type

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@register_jitable
def parse_quoted_string(path: str, i: int) -> Tuple[int, str, str]:  # pragma: no cover:
    """Parse a string starting at position i

    Note that both single and double quoted strings are suppported.

    Args:
        path: the path being parsed
        i: the position to start scanning for a quoted string

    Returns:
        Tuple[int, str, str]: a tuple containing the position of the parser
        after parsing an integer and the string that was parsed, and any errors
        encountered. If the error string has a non-zero length, then the first
        two fields should not be read.
    """

    if path[i] != '"' and path[i] != "'":
        return (-1, "", "Expected quoted string")
    quote = path[i]

    value = ""
    i += 1
    escaping = False
    closed = False
    while i < len(path):
        if escaping:
            escaping = False
            value += path[i]
        else:
            if path[i] == "\\":
                escaping = True
            elif path[i] == quote:
                closed = True
            else:
                value += path[i]
        i += 1
        if closed:
            break
    if not closed:
        return (-1, "", "Unterminated quoted string argument")
    if len(value) == 0:
        return (-1, "", "Unexpected empty quoted string")
    return i, value, ""


@register_jitable
def parse_int(path: str, i: int) -> Tuple[int, int, str]:  # pragma: no cover
    """Parse an integer starting at position i
    Args:
        path: the path being parsed
        i: the position to start scanning for an integer

    Returns:
        Tuple[int, int, str]: a tuple containing the position of the parser after
    parsing an integer, the integer that was parsed, and an error message. If the er
    """
    value = 0
    char0 = ord("0")
    if not path[i].isdigit():
        return (
            -1,
            -1,
            f"Expected digit, but got '{path[i]}' at position {i} of {path}",
        )
    while path[i].isdigit():
        value *= 10
        value += ord(path[i]) - char0
        i += 1
    return i, value, ""


@register_jitable
def process_json_path(
    path: str,
) -> Tuple[List[Tuple[int, str]], str]:  # pragma: no cover
    """Utility for json_extract_path_text_util to take in a path and break it
       up into each component index/field.

    Args:
        path (string): the path being used to parse a JSON string.

    Returns:
        tuple[list[tuple[int, string]], string]: a tuple with the result of the
        operation and an error message. If the error message is empty, the
        result will be the components of a path, each with a valid index value
        if it was an index operation, and a field name if not. If the component
        was a field name, the integer will always be -1. If the error message
        is non-empty, then the result should not be read.

    For example:

    path = '[3].Person["Name"].First'
    Returns the following list: [(3, ""), (-1, "Person"), (-1, "Name"), (-1, "First")]
    """

    path_parts = []

    # If this boolean flag is set True at any point, it means that the path is
    # malformed so an exception will be raised
    if len(path) == 0:
        return (path_parts, "Expected non-empty path")

    # Keep scanning until the end of the path is reached or the invalid flag
    # is set to True
    i = 0
    while i < len(path):
        if path[i] == ".":
            return (path_parts, "Unexpected '.' in JSON path argument")
        str_arg = ""
        int_arg = -1
        if path[i] == "[":
            i += 1
            if i == len(path):
                return (
                    path_parts,
                    f"Expected index but got END OF STRING at position {i} in JSON path argument",
                )
            if path[i] == '"' or path[i] == "'":
                # Accessing a field via []
                i, str_arg, err_msg = parse_quoted_string(path, i)
                if len(err_msg) > 0:
                    return (path_parts, err_msg)
            else:
                # Indexing
                i, int_arg, err_msg = parse_int(path, i)
                if len(err_msg) > 0:
                    return (path_parts, err_msg)

            if i == len(path):
                return (
                    path_parts,
                    f"Expected ']' but got END OF STRING at position {i} in JSON path argument",
                )
            if path[i] != "]":
                return (
                    path_parts,
                    f"Expected ']' but got {path[i]} at position {i} in JSON path argument",
                )
            i += 1
        else:
            if path[i] == '"':
                # Parsing a quoted field
                i, str_arg, err_msg = parse_quoted_string(path, i)
                if len(err_msg) > 0:
                    return (path_parts, err_msg)
            else:
                # Parsing a normal field, consume all characters until '.' or '['
                start = i
                while i < len(path):
                    if path[i] == "[" or path[i] == ".":
                        break
                    i += 1
                str_arg = path[start:i]
        path_parts.append((int_arg, str_arg))

        # If this isn't the end of the string, we expect a . or a [
        if i != len(path):
            if path[i] != "." and path[i] != "[":
                return (
                    path_parts,
                    f"Expected one of '.', '[' but got {path[i]} at position {i} in JSON path argument",
                )
            if path[i] == ".":
                i += 1
                if i == len(path) or path[i] == "[":
                    return (path_parts, "Unexpected '.' in JSON path argument")

    return (path_parts, "")


@register_jitable
def consume_whitespace(data: str, pos: int) -> int:  # pragma: no cover
    """Return the position of the parser after consuming leading whitespace"""
    while data[pos].isspace():
        pos += 1
    return pos


@register_jitable
def consume_expected(data: str, pos: int, expected: str) -> int:  # pragma: no cover
    """Return the position of the parser after consuming an expected string"""
    actual = data[pos : pos + len(expected)]
    if actual != expected:
        raise ValueError(f"Expected {expected}, but got {actual}")
    return pos + len(expected)


@register_jitable
def consume_json_value(data: str, pos: int) -> int:  # pragma: no cover
    """Return the position of the parser after consuming a valid JSON value"""
    brace_stack = []
    quoted = False
    quote = None
    while pos < len(data):
        c = data[pos]
        if quoted:
            if c == quote:
                quoted = False
            elif c == "\\":
                if (pos + 1) == len(data):
                    raise ValueError("Expected escaped character but got end of string")
                pos += 1
            pos += 1
            continue

        if len(brace_stack) == 0 and (c == "," or c == "]" or c == "}"):
            return pos

        if c == '"' or c == "'":
            quoted = True
            quote = c
        elif c == "[" or c == "{":
            brace_stack.append(c)
        else:
            if (c == "]" and brace_stack[-1] == "[") or (
                c == "}" and brace_stack[-1] == "{"
            ):
                brace_stack.pop()
        pos += 1
    if len(brace_stack) or quoted:
        raise ValueError("Malformed JSON input")
    return pos


@register_jitable
def parse_and_extract_json_string(
    data: str, path: str
) -> Union[str, None]:  # pragma: no cover
    """Utility for json_extract_path_text_util to use on specific strings

    Args:
        data (string): the string being parsed and extracted
        path (string): the path to extract from

    Returns:
        string: see json_extract_path_text_util for specification
    """
    # Pre-process the path to make sure it is valid, and extract a list of
    # tuples indicating which paths are fields vs indices, and the indices
    # themselves.
    path_parts, error_msg = process_json_path(path)
    if len(error_msg) != 0:
        raise ValueError(error_msg)

    invalid_data = False

    # Repeat the procedure until we reach the end
    # while True:
    for target_index, target_key in path_parts:
        new_data = None
        i = consume_whitespace(data, 0)
        if target_index >= 0:
            if data[i] != "[":
                new_data = None
            else:
                i += 1
                # Consume the first `target_index` values
                for _ in range(target_index):
                    i = consume_whitespace(data, i)
                    i = consume_json_value(data, i)
                    i = consume_whitespace(data, i)
                    if data[i] == "]":
                        break
                    i = consume_expected(data, i, ",")

                if data[i] != "]":
                    # Get the current value and save it as new_data
                    i = consume_whitespace(data, i)
                    value_start = i
                    i = consume_json_value(data, i)
                    new_data = data[value_start:i]
        else:
            if data[i] != "{":
                return None
            i += 1
            while True:
                # Get the current key
                i = consume_whitespace(data, i)
                i, key, err_msg = parse_quoted_string(data, i)
                if len(err_msg) > 0:
                    raise ValueError(err_msg)

                i = consume_whitespace(data, i)
                i = consume_expected(data, i, ":")

                # Get the current value
                i = consume_whitespace(data, i)
                value_start = i
                i = consume_json_value(data, i)

                # If the current key is target_key, then save the current value
                # as new_data and break
                if key == target_key:
                    new_data = data[value_start:i]
                    break

                i = consume_whitespace(data, i)
                if data[i] == "}":
                    i += 1
                    break
                i = consume_expected(data, i, ",")

        if new_data is None:
            return None
        data = new_data.strip()

    if invalid_data:
        raise ValueError(f"JSON extraction: malformed string cannot be parsed")

    # Special case for strings - we parse the string into a SQL string
    if data[0] == '"' or data[0] == "'":
        ret_val = ""
        escaping = False
        for i in range(1, len(data) - 1):
            if not escaping and data[i] == "\\":
                escaping = True
            else:
                escaping = False
                ret_val += data[i]
        return ret_val
    return data


def get_path(data, path, is_scalar=False):  # pragma: no cover
    pass


@overload(get_path, no_unliteral=True)
def overload_get_path(data, path, is_scalar=False):  # pragma: no cover
    args = [data, path, is_scalar]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.get_path",
                ["data", "path", "is_scalar"],
                i,
                default_map={"is_scalar": False},
            )

    def impl(data, path, is_scalar=False):
        return bodo.libs.bodosql_json_array_kernels.get_path_util(data, path, is_scalar)

    return impl


def get_path_util(data, path, is_scalar):  # pragma: no cover
    """dummy method for overload"""
    pass


@overload(get_path_util, no_unliteral=True)
def overload_get_path_util(data, path, is_scalar):
    if not is_overload_constant_str(path):
        raise_bodo_error("Only constant path expressions are supported at this time")

    is_scalar_bool = get_overload_const_bool(is_scalar)

    path_parts, err_msg = process_json_path(get_overload_const_str(path))
    if len(err_msg) > 0:
        raise ValueError(err_msg)

    func_text = "def impl(data, path, is_scalar):\n"
    func_text += f"  tmp0 = data\n"
    for i, (target_index, target_key) in enumerate(path_parts):
        if target_index >= 0:
            func_text += f"  tmp{i + 1} = bodo.libs.bodosql_array_kernels.arr_get(tmp{i}, {target_index}, is_scalar_arr={is_scalar_bool})\n"
        else:
            func_text += f"  tmp{i + 1} = bodo.libs.bodosql_json_array_kernels.get_field(tmp{i}, {repr(target_key)}, {is_scalar_bool})\n"
    func_text += f"  return tmp{len(path_parts)}\n"

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def get_field(data, key, is_scalar, ignore_case=False):  # pragma: no cover
    """dummy method for overload"""
    pass


@overload(get_field, no_unliteral=True)
def overload_get_field(data, field, is_scalar, ignore_case=False):  # pragma: no cover
    args = [data, field, is_scalar, ignore_case]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.get_field",
                ["data", "field", "is_scalar", "ignore_case"],
                i,
                default_map={"ignore_case": False},
            )

    def impl(data, field, is_scalar, ignore_case=False):
        return bodo.libs.bodosql_json_array_kernels.get_field_util(
            data, field, is_scalar, ignore_case
        )

    return impl


def get_field_util(data, field, is_scalar, ignore_case):  # pragma: no cover
    """dummy method for overload"""
    pass


@overload(get_field_util, no_unliteral=True)
def overload_get_field_util(data, field, is_scalar, ignore_case):
    """BodoSQL Implementation of accessing a field from a MAP type (map/struct/etc). If the input is not a map type, returns NULL"""
    json_type = data
    if bodo.hiframes.pd_series_ext.is_series_type(json_type):
        json_type = json_type.data

    arg_names = ["data", "field", "is_scalar", "ignore_case"]
    arg_types = [data, field, is_scalar, ignore_case]
    propagate_null = [True, True, False, False]

    ignore_case_bool = get_overload_const_bool(ignore_case)

    struct_mode = False
    map_mode = False
    if isinstance(
        json_type, (bodo.StructArrayType, bodo.libs.struct_arr_ext.StructType)
    ):
        struct_mode = True
    elif isinstance(
        json_type, (bodo.MapArrayType, bodo.libs.map_arr_ext.MapScalarType)
    ):
        map_mode = True

    scalar_text = ""
    if struct_mode:
        if not is_overload_constant_str(field):
            raise_bodo_error(
                "Only constant path expressions are supported at this time"
            )
        key_str = get_overload_const_str(field)
        if ignore_case_bool:
            key_str = key_str.lower()
        field_pos = None
        n_fields = len(json_type.data)
        for i in range(n_fields):
            name = json_type.names[i]
            if ignore_case_bool:
                if name.lower() == key_str:  # pragma: no cover
                    field_pos = i
                    # Set the key_str equal to the actual field name
                    # so that the indexing in the later scalar func text
                    # will work properly
                    key_str = name
                    break
            else:
                if name == key_str:  # pragma: no cover
                    field_pos = i
                    break
        if field_pos == None:
            out_dtype = bodo.null_array_type
            scalar_text += "bodo.libs.array_kernels.setna(res, i)\n"
        else:
            out_dtype = json_type.data[field_pos]
            scalar_text += f"if bodo.libs.struct_arr_ext.is_field_value_null(arg0, {repr(key_str)}):\n"
            scalar_text += f"  bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += f"else:\n"
            scalar_text += f"  res[i] = arg0[{repr(key_str)}]\n"
    elif map_mode:
        out_dtype = bodo.utils.typing.to_nullable_type(json_type.value_arr_type)
        scalar_text += "keys = list(arg0)\n"
        if ignore_case_bool:
            scalar_text += "lowered_keys = [key.lower() for key in keys]\n"
            scalar_text += "lowered_arg1 = arg1.lower()\n"
            # For loop is needed here as using lowered_keys.index(lowered_arg1)
            # with exception catching causes an error
            scalar_text += "index_value = -1\n"
            scalar_text += (
                "for iter_val_not_used_in_gen_vec in range(len(lowered_keys)):\n"
            )
            scalar_text += (
                "  if lowered_keys[iter_val_not_used_in_gen_vec] == lowered_arg1:\n"
            )
            scalar_text += "    index_value = iter_val_not_used_in_gen_vec\n"
            scalar_text += "    break\n"
            scalar_text += "if index_value >= 0:\n"
            scalar_text += "  value = arg0[keys[index_value]]\n"
            scalar_text += "  if value is None:\n"
            scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "  else:\n"
            scalar_text += "    res[i] = value\n"
            scalar_text += "else:\n"
            scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
        else:
            scalar_text += "if arg1 in keys:\n"
            scalar_text += "  value = arg0[arg1]\n"
            scalar_text += "  if value is None:\n"
            scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "  else:\n"
            scalar_text += "    res[i] = value\n"
            scalar_text += "else:\n"
            scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
    else:
        out_dtype = bodo.null_array_type
        scalar_text += "bodo.libs.array_kernels.setna(res, i)\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=[
            not get_overload_const_bool(is_scalar),
            bodo.utils.utils.is_array_typ(field),
            False,
            False,
        ],
    )


def object_insert(
    data, new_field_name, new_field_value, update, is_scalar=False
):  # pragma: no cover
    pass


@overload(object_insert, no_unliteral=True)
def overload_object_insert(
    data, new_field_name, new_field_value, update, is_scalar=False
):  # pragma: no cover
    args = [data, new_field_name, new_field_value, update, is_scalar]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            if i == 2:
                # Ignore new_field_value: it's optional handling will be
                # done inside the kernel.
                continue
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.object_insert",
                ["data", "new_field_name", "new_field_value", "update", "is_scalar"],
                i,
                default_map={"is_scalar": False},
            )

    def impl(
        data, new_field_name, new_field_value, update, is_scalar=False
    ):  # pragma: no cover
        return bodo.libs.bodosql_json_array_kernels.object_insert_util(
            data, new_field_name, new_field_value, update, is_scalar
        )

    return impl


def object_insert_util(
    data, new_field_name, new_field_value, update, is_scalar
):  # pragma: no cover
    pass


@overload(object_insert_util, no_unliteral=True)
def overload_object_insert_util(
    data, new_field_name, new_field_value, update, is_scalar
):
    json_type = data
    if bodo.hiframes.pd_series_ext.is_series_type(json_type):
        json_type = json_type.data

    # If the input is null, just return it.
    if json_type == bodo.none or json_type == bodo.null_array_type:
        return (
            lambda data, new_field_name, new_field_value, update, is_scalar: data
        )  # pragma: no cover

    arg_names = ["data", "new_field_name", "new_field_value", "update", "is_scalar"]
    arg_types = [data, new_field_name, new_field_value, update, is_scalar]
    propagate_null = [True, True, False, True, False]
    extra_globals = {}
    scalar_text = ""
    is_scalar_bool = get_overload_const_bool(is_scalar)

    # If the value to insert is optional, unwrap the type and treat it normally
    # except that we need an un-optionalizing step in the runtime & a different
    # null check. This is done so we don't need to unify a struct where one
    # of the array types is sometimes XXX and sometimes a null array type.
    is_optional = False
    if isinstance(new_field_value, types.Optional):
        is_optional = True
        new_field_value = new_field_value.type

    are_arrays = [
        bodo.utils.utils.is_array_typ(data),
        bodo.utils.utils.is_array_typ(new_field_name),
        not is_scalar_bool,
        False,
        False,
    ]

    if isinstance(
        json_type, (bodo.StructArrayType, bodo.libs.struct_arr_ext.StructType)
    ):
        # Codepath for calling OBJECT_INSERT on a STRUCT.
        if bodo.hiframes.pd_series_ext.is_series_type(new_field_value):
            new_field_value = new_field_value.data
        if new_field_name == bodo.none:  # pragma: no cover
            new_field_name_str = ""
        else:
            if not is_overload_constant_str(new_field_name):  # pragma: no cover
                raise_bodo_error(
                    "object_insert unsupported on struct arrays with non-constant keys"
                )
            new_field_name_str = get_overload_const_str(new_field_name)

        if not is_overload_constant_bool(update):  # pragma: no cover
            raise_bodo_error("object_insert unsupported with non-constant update flag")
        update_bool = get_overload_const_bool(update)

        # Determine whether the newly injected field is null.
        inserting_null = (
            new_field_value == bodo.none or new_field_value == bodo.null_array_type
        )
        if inserting_null:
            new_null_check = "True"
        elif is_optional:
            new_null_check = "arg2 is None"
        elif is_scalar_bool:
            new_null_check = "False"
        else:
            new_null_check = "(bodo.libs.array_kernels.isna(new_field_value, i))"
        new_field_dtype = (
            new_field_value
            if not is_scalar_bool
            else bodo.utils.typing.to_nullable_type(
                bodo.utils.typing.dtype_to_array_type(new_field_value)
            )
        )

        # Keep track of the original struct field types as array types.
        original_types = list(json_type.data)
        if isinstance(data, bodo.StructType):
            original_types = [
                bodo.utils.typing.to_nullable_type(
                    bodo.utils.typing.dtype_to_array_type(typ)
                )
                for typ in original_types
            ]

        # The string representations of the new struct field values,
        values = []
        # The string representations of how to determine which struct fields are null.
        nulls = []
        # The new struct's field names.
        names = []
        # The new struct's inner array dtypes.
        dtypes = []

        # Iterate through all of the fields and either copy them over or
        # replace them with the newly injected field.
        n_fields = len(json_type.data)
        already_matched = False
        for i in range(n_fields):
            name = json_type.names[i]
            if name == new_field_name_str:
                if update_bool:
                    names.append(new_field_name_str)
                    nulls.append(new_null_check)
                    if (
                        new_field_value == bodo.none
                        or new_field_value == bodo.null_array_type
                    ):
                        # If replacing with null, keep the same dtype as before.
                        dtypes.append(original_types[i])
                        values.append(f"arg0['{name}']")
                    else:
                        # Otherwise, add the replacement.
                        dtypes.append(new_field_dtype)
                        values.append(f"None if {new_null_check} else arg2")
                    already_matched = True
                    continue
                else:
                    raise_bodo_error(
                        f"object_insert encountered duplicate field key '{name}'"
                    )

            # If the field is not being replaced, copy it over.
            null_check = f"bodo.libs.struct_arr_ext.is_field_value_null(arg0, '{name}')"
            names.append(name)
            nulls.append(null_check)
            values.append(f"arg0['{name}']")
            dtypes.append(original_types[i])

        # If the field was not already injected as a replacement, add it
        # to the end.
        if not already_matched:
            names.append(new_field_name_str)
            nulls.append(new_null_check)
            values.append(f"None if {null_check} else arg2")
            dtypes.append(new_field_dtype)

        # Use the null strings to create an array indicating which fields are null,
        # then create the new struct.
        extra_globals["names"] = bodo.utils.typing.ColNamesMetaType(tuple(names))
        scalar_text += f"null_vector = np.array([{', '.join(nulls)}], dtype=np.bool_)\n"
        scalar_text += f"res[i] = bodo.libs.struct_arr_ext.init_struct_with_nulls(({', '.join(values)},), null_vector, names)"
        out_dtype = bodo.StructArrayType(tuple(dtypes), tuple(names))

    elif isinstance(json_type, (bodo.MapArrayType, bodo.MapScalarType)):
        # Codepath for calling OBJECT_INSERT on a MAP.
        key_type = json_type.key_arr_type
        val_type = json_type.value_arr_type
        val_dtype = val_type.dtype

        new_field_dtype = (
            new_field_value
            if not bodo.utils.utils.is_array_typ(new_field_value, False)
            and not bodo.hiframes.pd_series_ext.is_series_type(new_field_value)
            else new_field_value.dtype
        )
        if new_field_dtype == bodo.none:  # pragma: no cover
            new_field_dtype = val_dtype

        types_to_unify = [val_dtype, new_field_dtype]
        common_dtype, _ = bodo.utils.typing.get_common_scalar_dtype(types_to_unify)
        if common_dtype == None:  # pragma: no cover
            raise_bodo_error("Incompatible value type for object_insert")
        common_dtype = bodo.utils.typing.dtype_to_array_type(common_dtype)
        if any(
            bodo.utils.typing.is_nullable(typ) for typ in [val_type, new_field_value]
        ):
            common_dtype = bodo.utils.typing.to_nullable_type(common_dtype)

        out_dtype = bodo.MapArrayType(key_type, common_dtype)

        extra_globals["struct_typ_tuple"] = (key_type, common_dtype)
        extra_globals["map_struct_names"] = bodo.utils.typing.ColNamesMetaType(
            ("key", "value")
        )

        # Only throw a duplicate field key exception if the update flag is not set
        scalar_text += "n_original_keys = len(arg0._keys)\n"
        scalar_text += "n_keys = n_original_keys + 1\n"
        scalar_text += "key_exists = False\n"
        scalar_text += "for idx in range(n_original_keys):\n"
        scalar_text += "  if arg0._keys[idx] == arg1:\n"
        scalar_text += "    if not arg3:\n"
        scalar_text += "        raise bodo.utils.typing.BodoError('object_insert encountered duplicate field key ' + arg1)\n"
        scalar_text += "    n_keys -= 1\n"
        scalar_text += "    key_exists = True\n"
        scalar_text += "    break\n"
        scalar_text += "struct_arr = bodo.libs.struct_arr_ext.pre_alloc_struct_array(n_keys, (-1,), struct_typ_tuple, ('key', 'value'), None)\n"

        # Copy over all the existing key value pairs, except any that match the update key.
        scalar_text += "write_idx = 0\n"
        scalar_text += "for read_idx in range(n_original_keys):\n"
        scalar_text += "  key = arg0._keys[read_idx]\n"

        # We know that this will only happen if update (arg3) is false. Otherwise it would have been caught above.
        scalar_text += "  if key == arg1:\n"
        if bodo.utils.utils.is_array_typ(new_field_value, True):
            scalar_text += (
                "    val_is_null = bodo.libs.array_kernels.isna(new_field_value, i)\n"
            )
        else:
            scalar_text += "    val_is_null = new_field_value is None\n"
        scalar_text += "    struct_arr[write_idx] = bodo.libs.struct_arr_ext.init_struct_with_nulls((arg1, arg2), (False, val_is_null), map_struct_names)\n"
        scalar_text += "  else:\n"
        scalar_text += "    value = arg0._values[read_idx]\n"
        scalar_text += (
            "    val_is_null = bodo.libs.array_kernels.isna(arg0._values, read_idx)\n"
        )
        scalar_text += "    struct_arr[write_idx] = bodo.libs.struct_arr_ext.init_struct_with_nulls((key, value), (False, val_is_null), map_struct_names)\n"
        scalar_text += "  write_idx += 1\n"
        # If it was not already added, insert the new key value pair
        scalar_text += "if not key_exists:\n"
        if bodo.utils.utils.is_array_typ(new_field_value, True):
            scalar_text += (
                "  val_is_null = bodo.libs.array_kernels.isna(new_field_value, i)\n"
            )
        else:
            scalar_text += "  val_is_null = new_field_value is None\n"
        scalar_text += "  struct_arr[n_keys - 1] = bodo.libs.struct_arr_ext.init_struct_with_nulls((arg1, arg2), (False, val_is_null), map_struct_names)\n"

        if any(are_arrays):
            scalar_text += "res[i] = struct_arr"
        else:
            # If our output should be scalar, then construct the map scalar we need
            scalar_text += (
                "key_data, value_data = bodo.libs.struct_arr_ext.get_data(struct_arr)\n"
            )
            scalar_text += (
                "nulls = bodo.libs.struct_arr_ext.get_null_bitmap(struct_arr)\n"
            )
            scalar_text += "res[i] = bodo.libs.map_arr_ext.init_map_value(key_data, value_data, nulls)\n"
    else:  # pragma: no cover
        raise_bodo_error(f"object_insert: unsupported type for json data '{json_type}'")

    # Avoid allocating dictionary-encoded output in gen_vectorized since not supported
    # by the kernel
    out_dtype = to_str_arr_if_dict_array(out_dtype)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
        are_arrays=are_arrays,
    )
