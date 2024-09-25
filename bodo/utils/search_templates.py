# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Determines amount of coverage based on a list of APIs. This list of API's is assumed
to be a text file where each line is a path which includes everything after the library 
for example: "Series.str.decode\n read_csv\n"
    usage: python -m bodo.utils.search_templates <path_to_apis>
"""
import sys
import warnings

import numba
import pandas as pd  # noqa
from numba import types
from numba.core.target_extension import dispatcher_registry, get_local_target
from numba.core.utils import order_by_target_specificity

import bodo
from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType
from bodo.hiframes.pd_index_ext import (
    MultiIndexType,
)
from bodo.hiframes.pd_offsets_ext import (
    DateOffsetType,
    MonthBeginType,
    MonthEndType,
    WeekType,
)
from bodo.hiframes.pd_rolling_ext import RollingType
from bodo.ir.unsupported_method_template import _UnsupportedTemplate

# Exception message indicating an attribute has an unsupported overload
unsupported_overload_exception_msg = "not supported yet"

# Define list of bodo types and how they relate to pandas
int_arr_typ = bodo.IntegerArrayType(bodo.int64)

series_types = [
    bodo.SeriesType(int_arr_typ),
    bodo.SeriesType(bodo.string_array_type),
    bodo.SeriesType(types.NPDatetime("ns")),
    bodo.SeriesType(types.NPTimedelta("ns")),
    bodo.SeriesType(bodo.StructArrayType((int_arr_typ,))),
    bodo.SeriesType(bodo.PDCategoricalDtype(None, None, None, data=int_arr_typ)),
]

dataframe_types = [bodo.DataFrameType(data=(int_arr_typ,), columns=("A",))]

index_types = [
    bodo.NumericIndexType(bodo.int64),
    bodo.StringIndexType(),
    bodo.BinaryIndexType(),
]

# Only keeping track of Pandas types for now.
bodo_pd_types_dict = {
    "Series": series_types,
    "DataFrame": dataframe_types,
    # Index types
    "Index": index_types,
    "RangeIndex": [bodo.RangeIndexType()],
    "IntervalIndex": [
        bodo.IntervalIndexType(bodo.IntervalArrayType(int_arr_typ)),
    ],
    "CategoricalIndex": [
        bodo.CategoricalIndexType(bodo.CategoricalArrayType(bodo.int64)),
    ],
    "MultiIndex": [MultiIndexType([int_arr_typ])],
    "DatetimeIndex": [
        bodo.DatetimeIndexType(),
    ],
    "TimedeltaIndex": [bodo.TimedeltaIndexType()],
    "PeriodIndex": [
        bodo.PeriodIndexType(1),
    ],
    # scalar/array types
    "Timestamp": [bodo.PandasTimestampType(1)],
    "Timedelta": [bodo.pd_timedelta_type],
    # dateoffset, dates
    "DateOffset": [DateOffsetType()],
    "MonthEnd": [MonthEndType()],
    "MonthBegin": [MonthBeginType()],
    "Week": [WeekType()],
    # window/groupby
    "Rolling": [RollingType(series_types[0], None, None, None)],
    # SeriesGroupBy type and DataFrameGroupbyType are the same for now
    "SeriesGroupBy": [
        DataFrameGroupByType(series_type, None, None, None)
        for series_type in series_types
    ],
    "DataFrameGroupBy": [
        DataFrameGroupByType(dataframe_type, None, None, None)
        for dataframe_type in dataframe_types
    ],
}


def is_attr_supported(typing_ctx, typ, attr):
    """Check if an specific attribute or method is supported for the given type.
    Args:
        typing_ctx: The context containing function/method/attribute templates to search.
        typ: The object to check.
        attr: The attribute to check.
    Returns:
        Boolean | None: True if there is an implementation of the specified method or
        attribute, False if there is an unsupported implementation. None if there is
        no implementation at all or it is unclear.
    """
    templates = list(typing_ctx._get_attribute_templates(typ))

    # get the order in which to try templates
    # TODO: can we delete these 2 lines and just use templates ?
    target_hw = get_local_target(typing_ctx)
    order = order_by_target_specificity(target_hw, templates, fnkey=attr)

    # flag that gets set the first time a matching template is encountered.
    # ensure that no method/attribute has both a supported template and an
    # unsupported template simultaneously.
    is_supported = None
    for template in order:
        if isinstance(template, _UnsupportedTemplate):
            is_matching_template = template.is_matching_template(attr)
        else:
            is_matching_template = template.resolve(typ, attr) is not None

        if is_matching_template and is_supported is None:
            is_supported = not isinstance(template, _UnsupportedTemplate)
        elif is_matching_template:
            # check that other templates defined are all supported or all unsupported
            curr_template_supported = not isinstance(template, _UnsupportedTemplate)
            if curr_template_supported != is_supported:
                return None

    # There is no template found (unsupported or otherwise)
    return is_supported


def lookup_template(typing_ctx, typ, path):
    """Search for a method or attribute starting from typ and following path
    For example, to look up whether Series.str.decode was supported:
        ser_typ = bodo.SeriesType(bodo.string_array_type)
        result = lookup_template(ser_typ, ["str", "decode"])

    Args:
        typing_ctx: The context containing function/method/attribute templates to search
        typ: The object used as a starting point to find a method or attribute
        path (List[str]): The path to find the method/attribute starting from typ
    Returns:
        Boolean | None: True if there is an implementation of the specified method or
        attribute, False if there is an unsupported implementation. None if there is
        no implementation at all.
    """
    for attr in path:
        is_supported = is_attr_supported(typing_ctx, typ, attr)
        if is_supported:
            typ = typing_ctx.resolve_getattr(typ, attr)
        else:
            # False (as in Unsupported template was found) or None
            return is_supported
    return True


def generate_is_supported_str(path, is_supported, no_overload_warn_message):
    """Generates a string for purposes of creating coverage spreadsheets."""
    supported_str = path
    if is_supported is None:
        return path + ", No, " + no_overload_warn_message
    elif is_supported:
        return supported_str + ", YES"
    else:
        return supported_str + ", NO"


def lookup_all(paths, typing_ctx, types_dict, keys=None, print_out=False):
    """Lookup a list of apis e.g. ["Series.str.split", "read_csv", "DataFrame.cut", ...]
    from a library and determine the coverage for each of them. Gives a warning
    When a method or attribute is unsupported and does not have proper error messaging.

    Args:
        paths (List[str]): a list of paths.
        typing_ctx: The context containing function/method/attribute templates to search.
        types_dict (Dict[str, List]): Mapping from type name to instances of that type.
        keys (List[str]): List of types in types_dict to include in the search.
        print_out (Boolean): Whether to print each "{path_name}, {YES / NO}"
    """
    num_warnings = 0
    num_supported = 0
    num_unsupported_w_overload = 0
    no_overload_warn_message = (
        "No unsupported overload template found or conflicting overload templates."
    )

    # filter types_dict using keys, ignore keys not in types dict types_dict
    if keys is not None:
        keys = filter(lambda key: key in types_dict, keys)
        types_dict = {key: types_dict[key] for key in keys}

    for path in paths:
        path_list = path.split(".")
        base = path_list[0]

        # catch all message for types that don't have a bodo equivalent
        is_supported = None

        if base in types_dict:
            all_exceptions = []
            for typ in types_dict[base]:
                try:
                    is_supported = lookup_template(typing_ctx, typ, path_list[1:])
                    break
                except Exception as e:
                    # TODO: check exceptions list
                    all_exceptions.append(e)
        else:
            # Type/module not in the list of types we are currently checking
            continue
        # manually keep track of supported types with bodo equivalent
        if len(path_list) == 1 and base in types_dict:
            is_supported = True

        # No supported or unsupported overload template was found
        if is_supported is None:
            warnings.warn(path + ": " + no_overload_warn_message)
            num_warnings += 1
        elif is_supported:
            num_supported += 1
        else:
            num_unsupported_w_overload += 1

        if print_out:
            supported_str = generate_is_supported_str(
                path, is_supported, no_overload_warn_message
            )
            print(path + ", " + supported_str)

    warnings.warn(
        f"Found {num_supported} methods and attributes in the Pandas documentation that are currently supported. {num_unsupported_w_overload} that are unsupported but have an unsupported overload. {num_warnings} that are unsupported with no unsupported overload or have conflicting overloads."
    )


def main(args):
    path_to_apis = args[1]
    with open(path_to_apis, "r") as f:
        pandas_apis = f.readlines()
        pandas_apis = list(map(lambda x: x.strip(), pandas_apis))

    # We only target the CPU. The other option is Numba ufuncs
    disp = dispatcher_registry[numba.core.target_extension.CPU]
    typing_ctx = disp.targetdescr.typing_context
    # Probably not necessary to refresh
    typing_ctx.refresh()
    lookup_all(
        pandas_apis,
        typing_ctx,
        types_dict=bodo_pd_types_dict,
        keys=None,
        print_out=True,
    )


if __name__ == "__main__":
    main(sys.argv)
