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
def process_json_path(path: str) -> List[Tuple[bool, str]]:  # pragma: no cover
    """Utility for json_extract_path_text_util to take in a path and break it
       up into each component index/field, and identify which of the two it is.

    Args:
        path (string): the path being used to parse a JSON string.

    Returns:
        list[tuple[boolean, string]]: the components of a path, each with a boolean
        that is True if the component is an index, and False if it is a field name.

    Raises:
        ValueError: if the path is malformed

    For example:

    path = '[3].Person["Name"].First'
    Returns the following list: [(True, "3"), (False, "Person"), (False, "Name"), (False, "FIrst")]
    """

    # If this boolean flag is set True at any point, it means that the path is
    # malformed so an exception will be raised
    invalid_path = False
    if len(path) == 0:
        invalid_path = True

    path_parts = []

    # Keep scanning until the end of the path is reached or the invalid flag
    # is set to True
    i = 0
    while i < len(path):
        if invalid_path:
            break

        # Handling sections of the path in the form [index] or ["field_name"]
        if path[i] == "[":
            # Various boolean flags used to keep track of the characters inside
            # of the brackets:
            # - found_string: if True, then the component is of the form ["field_name"]
            # - found_number: if True, then the component is of the form [index]
            # - finished_value: if True, then the component is completed and searching
            #                   for a right bracket to terminate it
            # - found_key: if True, then the right bracket has been found
            # - escaped: should the next character be escaped
            # - in_str: is a string currently being processed
            # - string_type: if in_str is True, which quote type started the string
            # - current_part: accumulator of characters in the current component,
            #                 exclusing the brackets and quotes (if present)
            found_string = False
            found_number = False
            finished_value = False
            found_key = False
            escaped = False
            in_str = False
            string_type = ""
            current_part = ""

            # Scan through the remaining characters after the left bracket
            for j in range(i + 1, len(path)):
                # Handle cases where a string is being processed
                if in_str:
                    if escaped:
                        escaped = False
                        current_part += path[j]
                    elif path[j] == "\\":
                        escaped = True
                    elif path[j] == string_type:
                        finished_value = True
                        in_str = False
                    else:
                        current_part += path[j]

                # If not in a string, then ignore whitespace
                elif path[j].isspace():
                    if found_number:
                        finished_value = True

                # Handle the end of the current component by appending
                # to the final list and moving up the start pointer
                elif path[j] == "]":
                    if not found_number and not finished_value:
                        invalid_path = True
                        break
                    found_key = True
                    if found_string:
                        path_parts.append((False, current_part))
                    else:
                        # Verify that the index is a positive number:
                        current_part = current_part.strip()
                        if not current_part.isdigit() or int(current_part) < 0:
                            invalid_path = True
                            break
                        path_parts.append((True, current_part))
                    # If the next character is a dot, skip it
                    if j < len(path) - 2 and path[j + 1] == ".":
                        i = j + 2
                        break
                    # Otherwise, move up the start pointer to the next character
                    else:
                        i = j + 1
                        break

                # If the value has been finished and one of the prior cases
                # was not triggered, then the path is malformed
                elif finished_value:
                    invalid_path = True
                    break

                # If currently scanning a number, add the current character to it
                # unless it is a whitespace, which would mean the number is finished
                elif found_number:
                    if path[j].isspace():
                        finished_value = True
                    elif path[j].isdigit():
                        current_part += path[j]
                    else:
                        invalid_path = True
                        break

                # Handle cases where the left bracket has been found and
                # the start of the component is still being searched for
                else:
                    if path[j] in "'\"":
                        string_type = path[j]
                        in_str = True
                        found_string = True
                    elif path[j].isdigit():
                        found_number = True
                        current_part = path[j]
                    else:
                        invalid_path = True

            if not found_key:
                invalid_path = True
                break

        # Handling field names that are not enclosed by brackets and quotes
        else:
            # The first character in the field name must be a letter
            if not path[i].isalpha():
                invalid_path = True
                break

            # If the current letter is the last character in the path, then
            # it is the entire field name
            current_part = path[i]
            if i == len(path) - 1:
                path_parts.append((False, current_part))
                break

            # Loop through the remaining characters in the path until the end
            # of the current field name is found. If the end of the string is
            # encountered first, then halt the entire process
            found_ending = False
            for j in range(i + 1, len(path)):
                # If the current character is a dot, then everything from i
                # up to this point is the field name. If this is the end of
                # the string, or the next character is a bracket, then it is
                # malformed. The next field starts at the next character
                if path[j] == ".":
                    if j == len(path) - 1 or not path[j + 1].isalpha():
                        invalid_path = True
                        break
                    path_parts.append((False, current_part))
                    i = j + 1
                    break

                # If the current character is a left bracket, then everything
                # from i up to this point is the field name. The next field starts
                # at this character
                elif path[j] == "[":
                    path_parts.append((False, current_part))
                    i = j
                    break

                # Otherwise, append the current character to the field name
                else:
                    current_part += path[j]

                # If this is the end of the string, then the field name is
                # complete
                if j == len(path) - 1:
                    path_parts.append((False, current_part))
                    i = j + 1
                    found_ending = True

            if found_ending:
                break

    if invalid_path:
        raise ValueError("JSON extraction: Invalid path")

    return path_parts


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
    path_parts = process_json_path(path)

    invalid_data = False

    # Repeat the procedure until we reach the end
    # while True:
    for indexing, current_path in path_parts:
        # Various indices used to keep track of the distinct entities
        # in the string as they are being parsed, as well as the total
        # number of entries found, a stack of unclosed brackets, the opening
        # character of an unclosed string, whether the next character
        # is to be escaped, and whether the next value to be parsed is the
        # target value.
        current_key = ""
        current_value = ""
        entries_found = 0
        stack = ["[" if indexing else "{"]
        string_type = ""
        escaped = False
        new_data = None
        target_value = indexing and int(current_path) == 0

        # Initialize the state based on whether we are indexing into an array
        # or looking for a field in an object.
        state = 0 if indexing else 1

        for i in range(len(data)):
            # 0: look for the start of an array (ignoring whitespace)
            if state == 0:
                if data[i].isspace():
                    continue
                elif data[i] == "[":
                    state = 5
                else:
                    return None

            # 1: look for the start of an object (ignoring whitespace)
            elif state == 1:
                if data[i].isspace():
                    continue
                elif data[i] == "{":
                    state = 2
                else:
                    return None

            # 2: look for the start of a key string (ignoring whitespace)
            elif state == 2:
                if data[i].isspace():
                    continue
                elif data[i] in "'\"":
                    string_type = data[i]
                    state = 3
                else:
                    invalid_data = True
                    break

            # 3: look for the end of a key string
            elif state == 3:
                if escaped:
                    current_key += data[i]
                    escaped = False
                elif data[i] == "\\":
                    escaped = True
                elif data[i] == string_type:
                    target_value = current_key == current_path
                    state = 4
                else:
                    current_key += data[i]

            # 4: look for a colon (ignoring whitespace)
            elif state == 4:
                if data[i].isspace():
                    continue
                elif data[i] == ":":
                    state = 5
                else:
                    invalid_data = True
                    break

            # 5: look for value to be parsed (ignoring whitespace)
            elif state == 5:
                if data[i].isspace():
                    continue
                elif data[i] in "'\"":
                    if target_value:
                        current_value += data[i]
                    string_type = data[i]
                    state = 7
                elif data[i] in "[{":
                    stack.append(data[i])
                    if target_value:
                        current_value += data[i]
                    state = 6
                else:
                    if target_value:
                        current_value += data[i]
                    state = 6

            # 6: parse a value
            elif state == 6:
                # If a comma is encountered and the top level of data is
                # being processed, then the current value has terminated
                if data[i] == "," and len(stack) == 1:
                    if indexing:
                        if target_value:
                            new_data = current_value.strip()
                            break
                        state = 5
                        entries_found += 1
                        target_value = int(current_path) == entries_found
                    else:
                        if target_value:
                            new_data = current_value.strip()
                            break
                        state = 2
                        current_key = ""

                # If a "]" is encountered and the top level of data is
                # being processed, then the current value has terminated. It
                # is the answer if the number of entries already found is the
                # index that is being sought
                elif indexing and data[i] == "]" and len(stack) == 1:
                    if entries_found == int(current_path):
                        new_data = current_value.strip()
                        break
                    return None

                # If a "}" is encountered and the top level of data is
                # being processed, then the current value has terminated. Its
                # value is the answer if the key matches the field name sought
                elif (not indexing) and data[i] == "}" and len(stack) == 1:
                    if current_key == current_path:
                        new_data = current_value.strip()
                        break
                    return None

                # If a single or double quote is encountered, a string has begun
                elif data[i] in "'\"":
                    if target_value:
                        current_value += data[i]
                    string_type = data[i]
                    state = 7

                # If a left bracket (square or curly) is encountered, push them
                # onto the stack
                elif data[i] in "[{":
                    if target_value:
                        current_value += data[i]
                    stack.append(data[i])

                # If a right bracket is encountered, make sure it matches the
                # top value of the stack, then pop the top element of the stack
                elif data[i] == "]":
                    if target_value:
                        current_value += data[i]
                    if len(stack) == 0 or stack[-1] != "[":
                        invalid_data = True
                        break
                    else:
                        stack.pop()
                elif data[i] == "}":
                    if target_value:
                        current_value += data[i]
                    if len(stack) == 0 or stack[-1] != "{":
                        invalid_data = True
                        break
                    else:
                        stack.pop()

                # Otherwise, append the current character to the value
                else:
                    if target_value:
                        current_value += data[i]

            # 7: parse a substring
            elif state == 7:
                if escaped:
                    if target_value:
                        current_value += data[i]
                    escaped = False
                elif data[i] == "\\":
                    escaped = True
                elif data[i] == string_type:
                    if target_value:
                        current_value += data[i]
                    state = 6
                else:
                    if target_value:
                        current_value += data[i]

        if invalid_data:
            break

        if new_data is None:
            return None
        else:
            if new_data[0] in "'\"":
                if len(new_data) < 2 or new_data[-1] != new_data[0]:
                    invalid_data = True
                    break
                else:
                    new_data = new_data[1:-1]

            data = new_data

    if invalid_data:
        raise ValueError(f"JSON extraction: malformed string cannot be parsed")

    return data


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
        if i == 2:
            # ignore new_field_value
            continue
        if isinstance(args[i], types.optional):  # pragma: no cover
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


@numba.generated_jit(nopython=True)
def object_insert_util(data, new_field_name, new_field_value, update, is_scalar):
    json_type = data
    if bodo.hiframes.pd_series_ext.is_series_type(json_type):
        json_type = json_type.data

    struct_mode = False
    map_mode = False
    none_mode = False
    if json_type == bodo.none:  # pragma: no cover
        none_mode = True
    elif isinstance(
        json_type, (bodo.StructArrayType, bodo.libs.struct_arr_ext.StructType)
    ):
        struct_mode = True
    elif isinstance(json_type, bodo.MapArrayType):
        map_mode = True
    else:  # pragma: no cover
        raise_bodo_error(f"object_insert: unsupported type for json data '{json_type}'")

    arg_names = ["data", "new_field_name", "new_field_value", "update", "is_scalar"]
    arg_types = [data, new_field_name, new_field_value, update, is_scalar]
    propagate_null = [True, True, False, True, False]
    extra_globals = {}
    scalar_text = ""

    if none_mode:  # pragma: no cover
        out_dtype = bodo.none
    elif struct_mode:
        if bodo.hiframes.pd_series_ext.is_series_type(new_field_value):
            new_field_value = new_field_value.data

        if new_field_name == bodo.none:  # pragma: no cover
            new_field_name_str = ""
        else:
            if not is_overload_constant_str(new_field_name):  # pragma: no cover
                # TODO improve error msg
                raise_bodo_error(
                    "object_insert unsupported on struct arrays with non-constant keys"
                )
            new_field_name_str = get_overload_const_str(new_field_name)

        if not is_overload_constant_bool(update):  # pragma: no cover
            raise_bodo_error(
                "object_insert unsupported on struct arrays with non-constant update flag"
            )
        update_bool = get_overload_const_bool(update)

        data = []
        nulls = []
        names = []
        dtypes = []
        n_fields = len(json_type.data)
        for i in range(n_fields):
            name = json_type.names[i]
            if name == new_field_name_str:  # pragma: no cover
                if update_bool:
                    continue
                else:
                    raise_bodo_error(
                        f"object_insert encountered duplicate field key '{name}'"
                    )
            null_check = f"bodo.libs.struct_arr_ext.is_field_value_null(arg0, '{name}')"
            names.append(name)
            nulls.append(null_check)
            data.append(f"None if {null_check} else arg0['{name}']")
            field_dtype = json_type.data[i]
            if not bodo.utils.utils.is_array_typ(json_type, False):  # pragma: no cover
                field_dtype = bodo.utils.typing.dtype_to_array_type(field_dtype)
            dtypes.append(field_dtype)

        names.append(new_field_name_str)
        if new_field_value == bodo.none:  # pragma: no cover
            new_field_value = bodo.optional(bodo.int64)
            nulls.append("True")
            data.append(f"None")
        else:
            nulls.append("(arg2 is None)")
            data.append(f"None if {nulls[-1]} else arg2")

        new_field_dtype = (
            new_field_value
            if not get_overload_const_bool(is_scalar)
            else bodo.utils.typing.dtype_to_array_type(new_field_value)
        )
        dtypes.append(new_field_dtype)

        extra_globals["names"] = bodo.utils.typing.ColNamesMetaType(tuple(names))
        scalar_text += f"null_vector = np.array([{', '.join(nulls)}], dtype=np.bool_)\n"
        scalar_text += f"res[i] = bodo.libs.struct_arr_ext.init_struct_with_nulls(({', '.join(data)},), null_vector, names)"
        out_dtype = bodo.StructArrayType(tuple(dtypes), tuple(names))
    else:
        if map_mode:
            key_type = json_type.key_arr_type
            val_type = json_type.value_arr_type
            val_dtype = val_type.dtype
        else:  # pragma: no cover
            key_type = json_type.key_type
            val_type = json_type.value_type
            val_dtype = val_type

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

        scalar_text += "keys = list(arg0)\n"
        # Only throw a duplicate field key exception if the update flag is not set
        scalar_text += "n_keys = len(keys) + 1\n"
        scalar_text += "if arg1 in keys:\n"
        scalar_text += "  if not arg3:\n"
        scalar_text += "    raise bodo.utils.typing.BodoError('object_insert encountered duplicate field key ' + arg1)\n"
        scalar_text += "  n_keys -= 1\n"
        scalar_text += "struct_arr = bodo.libs.struct_arr_ext.pre_alloc_struct_array(n_keys, (-1,), struct_typ_tuple, ('key', 'value'))\n"

        scalar_text += "idx = 0\n"
        scalar_text += "for key in keys:\n"
        # We know that this will only happen if update (arg3) is false. Otherwise it would have been caught above.
        scalar_text += "  if key == arg1:\n"
        scalar_text += "    continue\n"
        scalar_text += "  value = arg0[key]\n"
        scalar_text += (
            "  in_offsets = bodo.libs.array_item_arr_ext.get_offsets(data._data)\n"
        )
        scalar_text += (
            "  in_struct_arr = bodo.libs.array_item_arr_ext.get_data(data._data)\n"
        )
        scalar_text += "  start_offset = in_offsets[np.int64(i)]\n"
        scalar_text += "  curr_struct = in_struct_arr[start_offset + idx]\n"
        scalar_text += "  val_is_null = bodo.libs.struct_arr_ext.is_field_value_null(curr_struct, 'value')\n"
        scalar_text += "  struct_arr[idx] = bodo.libs.struct_arr_ext.init_struct_with_nulls((key, value), (False, val_is_null), map_struct_names)\n"
        scalar_text += "  idx += 1\n"

        if bodo.utils.utils.is_array_typ(new_field_value, True):
            scalar_text += (
                "val_is_null = bodo.libs.array_kernels.isna(new_field_value, i)\n"
            )
        else:  # pragma: no cover
            scalar_text += "val_is_null = new_field_value is None\n"
        scalar_text += "struct_arr[n_keys - 1] = bodo.libs.struct_arr_ext.init_struct_with_nulls((arg1, arg2), (False, val_is_null), map_struct_names)\n"

        scalar_text += "res[i] = struct_arr"

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
        are_arrays=[
            (
                bodo.utils.utils.is_array_typ(typ)
                if i != 2
                else not get_overload_const_bool(is_scalar)
            )
            for i, typ in enumerate(arg_types)
        ],
    )
