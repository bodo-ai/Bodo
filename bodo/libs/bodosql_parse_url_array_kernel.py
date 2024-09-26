# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Implements the parse url array kernel, specific to BodoSQL. This is done in it's own file due to the number
of helper functions needed.
"""

import re

import numpy as np
from numba.core import types
from numba.extending import overload

import bodo
import bodo.libs.uuid
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.libs.str_arr_ext import str_arr_set_na
from bodo.utils.typing import is_overload_false


def parse_url(data, permissive_flag=False):
    pass


@overload(parse_url, no_unliteral=True)
def overload_parse_url(data, permissive_flag=False):  # pragma: no cover
    args = [data, permissive_flag]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.object_insert",
                ["data", "permissive_flag"],
                i,
                default_map={"permissive_flag": False},
            )

    def impl(data, permissive_flag=False):  # pragma: no cover
        return parse_url_util(data, permissive_flag)

    return impl


def parse_url_util(data, permissive_flag):  # pragma: no cover
    pass


@overload(parse_url_util, no_unliteral=True)
def overload_parse_url_util(data, permissive_flag):
    if not is_overload_constant_bool(permissive_flag):  # pragma: no cover
        raise_bodo_error("parse_url: permissive flag must be a constant")
    elif not is_overload_false(permissive_flag):  # pragma: no cover
        raise_bodo_error(
            "parse_url: invalid permissive flag. Only permissive=False is supported"
        )

    # This should be handled in BodoSQL. This is a sanity check.
    assert is_valid_string_arg(data), "Error in parse_url: data must be a string"

    field_names = ("fragment", "host", "parameters", "path", "port", "query", "scheme")
    child_types = (
        bodo.string_array_type,
        bodo.string_array_type,
        bodo.MapArrayType(bodo.string_array_type, bodo.string_array_type),
        bodo.string_array_type,
        bodo.string_array_type,
        bodo.string_array_type,
        bodo.string_array_type,
    )

    out_dtype = bodo.StructArrayType(tuple(child_types), field_names)
    arg_names = ["data", "permissive_flag"]
    arg_types = [data, permissive_flag]
    propagate_null = [True, False]

    scalar_text = "(scheme, netloc, path, query, fragment) = parse_url_wrapper(arg0)\n"
    scalar_text += "(host, port) = parse_netlock_into_host_and_port(netloc)\n"
    scalar_text += "query_parameters_as_map = parse_query_into_map(query)\n"
    # Reordered to match snowflake's ordering for my own sanity
    # NOTE: urlparse doesn't exactly line up with snowflake's behavior:
    # SF's scheme is always returned in uppercase, and SF omitts the first "/" in the path
    scalar_text += "if len(path) > 0 and path[0] == '/':\n"
    scalar_text += "    path = path[1:]\n"
    scalar_text += "struct_values = (fragment, host, query_parameters_as_map, path, port, query, scheme)\n"

    # Convert empty strings into null values to match snowflake behavior.
    scalar_text += f"null_vector = np.zeros(7, np.bool_)\n"
    scalar_text += "for substr, substr_idx in [(fragment, 0), (host, 1), (path, 3), (port, 4), (query, 5), (scheme, 6)]:\n"
    scalar_text += "    if substr == '':\n"
    scalar_text += "        null_vector[substr_idx] = True\n"
    # Convert empty dict into null value for query_parameters_as_map
    scalar_text += "if len(query_parameters_as_map._keys) == 0:\n"
    scalar_text += "    null_vector[2] = True\n"
    scalar_text += f"struct_output = bodo.libs.struct_arr_ext.init_struct_with_nulls(struct_values, null_vector, struct_names)\n"
    scalar_text += f"res[i] = struct_output\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals={
            "parse_query_into_map": parse_query_into_map,
            "parse_url_wrapper": parse_url_wrapper,
            "parse_netlock_into_host_and_port": parse_netlock_into_host_and_port,
            "struct_names": bodo.utils.typing.ColNamesMetaType(field_names),
        },
    )


# ____________________ Helper functions for parse_url ____________________


def parse_query_into_map(data):  # pragma: no cover
    pass


@overload(parse_query_into_map, no_unliteral=True)
def parse_query_into_map_overload(data):
    """
    Parses a query string of the form KEY1=VALUE1&KEY2=VALUE2&... into a dictionary.

    When handling malformed query strings, this helper function aproximates the observed
    behavior of the snowflake function PARSE_QUERY. IE:

    Ignore repeated &'s
    Repeated ?'s are treated as part of the key/value strings
    keys without values are assigned a value of null
    You can have a value without a key, the key will be empty string
    Repeated key's will be assigned the last occuring value
    The leftmost "=" is used to split key/value pairs. all Subsequent "="'s are assumed to be part of the name of the value.
    back/forward slashes are considered part of the key/value names

    Queries used to test Snowflake behavior:

    SELECT PARSE_URL('HTTPS://USER:PASS@EXAMPLE.INT:4345/HELLO.PHP?12345=1&&&&&ElectricBogalooHello=2ElectricBogalooHello=2')
    SELECT PARSE_URL('HTTPS://USER:PASS@EXAMPLE.INT:4345/HELLO.PHP?????12345&&&&&??ElectricBogalooHello')
    SELECT PARSE_URL('HTTPS://USER:PASS@EXAMPLE.INT:4345/HELLO.PHP?12345=1=1=1&&&&&??Electric=======BogalooHello')
    SELECT PARSE_URL('HTTPS://USER:PASS@EXAMPLE.INT:4345/HELLO.PHP?12345=1=\\\\\1=1&&&\\&&??Electric=====\\\\==BogalooHello')
    SELECT PARSE_URL('HTTPS://USER:PASS@EXAMPLE.INT:4345/HELLO.PHP?123///45=1=\\\\\1=1&&&\\&&?//?Ele///ctric=====\\\\==Bo///////gal//ooHello')
    SELECT PARSE_URL('HTTPS://USER:PASS@EXAMPLE.INT:4345/HELLO.PHP?&=2')
    SELECT PARSE_URL('HTTPS://USER:PASS@EXAMPLE.INT:4345/HELLO.PHP?hello=1&hello=2')
    """

    def impl(data):  # pragma: no cover
        keys_with_null_values = set()
        key_value_map = dict()

        for substr in data.split("&"):
            # Filter out empty substrings
            if substr == "":
                continue

            # Find the leftmost "="
            equality_idx = substr.find("=")
            if equality_idx != -1:
                key = substr[:equality_idx]
                value = substr[equality_idx + 1 :]
                if key in keys_with_null_values:
                    # remove from keys_with_null_values set to prevent double counting the same key
                    keys_with_null_values.remove(key)
                key_value_map[key] = value
            else:
                if substr in key_value_map:
                    # remove from key_value_map set to prevent double counting the same key
                    del key_value_map[substr]
                keys_with_null_values.add(substr)

        # Now that we know the number of key/value pairs, allocate the output arrays
        num_key_value_pairs = len(key_value_map) + len(keys_with_null_values)
        key_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(num_key_value_pairs, -1)
        value_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(
            num_key_value_pairs, -1
        )

        for idx, (key, value) in enumerate(key_value_map.items()):
            key_arr[idx] = key
            value_arr[idx] = value

        for idx, key in enumerate(keys_with_null_values):
            real_idx = idx + len(key_value_map)
            key_arr[real_idx] = key
            str_arr_set_na(value_arr, real_idx)

        # The fields in the struct are always present, so we don't need to set any null bits in
        # the map itself.
        null_bitmask = np.full((num_key_value_pairs + 7) >> 3, 255, np.uint8)

        output_map = bodo.libs.map_arr_ext.init_map_value(
            key_arr, value_arr, null_bitmask
        )
        return output_map

    return impl


def parse_url_wrapper(data):
    pass


@overload(parse_url_wrapper, no_unliteral=True)
def parse_url_wrapper_overload(data):
    """
    This is a function to handle the parsing of a url string.
    It is currently a wrapper around python's urlparse function. Eventually, we may want to
    replace this with a more efficient implementation, but this
    works for now.

    The expected return is a tuple of strings of the form
    (scheme, netloc, path, query, fragment)


    From a quick check, it seems that python's urlparse is EXTREMLY tolerant of malformed urls:


    In [2]: urlparse('HTTPS://UFIUYATFGHOU(*^&%$EW#ESDFXCGVHBJKO(U*&^%E$RDTFU(_&^%E%^DRYG&&3', scheme="
    ...: file")
    Out[2]: ParseResult(scheme='https', netloc='UFIUYATFGHOU(*^&%$EW', path='', params='', query='', fragment='ESDFXCGVHBJKO(U*&^%E$RDTFU(_&^%E%^DRYG&&3')

    In [3]: urlparse('HTT&*TYGUPO()&*T^RFTUYGIHOPUOY(T&*^RUF:???///::??//??:%%%$$$$PS://UFIUYATFGHOU(*^
    ...: &%$EW#ESDFXCGVHBJKO(U*&^%E$RDTFU(_&^%E%^DRYG&&3', scheme="file")
    Out[3]: ParseResult(scheme='file', netloc='', path='HTT&*TYGUPO()&*T^RFTUYGIHOPUOY(T&*^RUF:', params='', query='??///::??//??:%%%$$$$PS://UFIUYATFGHOU(*^&%$EW', fragment='ESDFXCGVHBJKO(U*&^%E$RDTFU(_&^%E%^DRYG&&3')

    In [4]: urlparse('HTT##&*TYGUPO()&*T^RFTUYGIHOPUOY(T&*^RUF:???///::??//??:%%%$$$$PS://UFIUYATFGHOU(
    ...: *^&%$EW#ESDFXCGVHBJKO(U*&^%E$RDTFU(_&^%E%^DRYG&&3', scheme="file")
    Out[4]: ParseResult(scheme='file', netloc='', path='HTT', params='', query='', fragment='#&*TYGUPO()&*T^RFTUYGIHOPUOY(T&*^RUF:???///::??//??:%%%$$$$PS://UFIUYATFGHOU(*^&%$EW#ESDFXCGVHBJKO(U*&^%E$RDTFU(_&^%E%^DRYG&&3')


    Therefore, this is very unlikely to raise an error in objmode.
    """
    from urllib.parse import urlparse

    def impl(data):  # pragma: no cover
        with bodo.objmode(
            scheme="string",
            netloc="string",
            path="string",
            params="string",
            query="string",
            fragment="string",
        ):
            named_tuple_out = urlparse(data)
            scheme = named_tuple_out.scheme
            netloc = named_tuple_out.netloc
            path = named_tuple_out.path
            params = named_tuple_out.params
            query = named_tuple_out.query
            fragment = named_tuple_out.fragment

        # url_parse's “parameter” section is a legacy thing from an older standard.
        # It's not used in modern URLs, so I've combined it with the path.
        # See the comments in this tickets for the full investigation:
        # https://bodo.atlassian.net/jira/software/c/projects/BSE/boards/25?assignee=60a3bbae47ba02006f1f8fbb&selectedIssue=BSE-1507
        if params != "":
            path += ";" + params
        return (scheme, netloc, path, query, fragment)

    return impl


def parse_netlock_into_host_and_port(netloc: str):  # pragma: no cover
    pass


@overload(parse_netlock_into_host_and_port, no_unliteral=True)
def parse_netlock_into_host_and_port_overload(netloc):
    def impl(netloc):  # pragma: no cover
        # There's probably a way to extract the port number directly
        # from the netloc string using regex in one expression,
        # but I'm not sure how to do that and this works in Bodo,
        # so I'm not going to push my luck.
        match = re.match(r".*:\d+\Z", netloc)
        if match is not None:
            port_splic_loc = netloc.rfind(":")
            return (netloc[:port_splic_loc], netloc[port_splic_loc + 1 :])
        else:
            return (netloc, "")

    return impl


# ____________________ End helper functions for parse_url ____________________
