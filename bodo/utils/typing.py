# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Helper functions to enable typing.
"""
import itertools
import operator
import types as pytypes
import warnings
from inspect import getfullargspec

import numba
import numba.cpython.unicode
import numpy as np
import pandas as pd
from numba.core import ir, ir_utils, types
from numba.core.errors import NumbaError
from numba.core.registry import CPUDispatcher
from numba.core.typing.templates import AbstractTemplate, signature
from numba.extending import (
    NativeValue,
    box,
    infer,
    intrinsic,
    lower_builtin,
    lower_cast,
    models,
    overload,
    overload_method,
    register_jitable,
    register_model,
    unbox,
)

import bodo

# sentinel string used in typing pass that specifies a const tuple as a const dict.
# const tuple is used since there is no literal type for dict
CONST_DICT_SENTINEL = "$_bodo_const_dict_$"


list_cumulative = {"cumsum", "cumprod", "cummin", "cummax"}


def is_timedelta_type(in_type):
    return in_type in [
        bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type,
        bodo.hiframes.datetime_date_ext.datetime_timedelta_type,
    ]


def is_dtype_nullable(in_dtype):
    """checks whether 'in_dtype' has sentinel NA values (as opposed to bitmap)"""
    return isinstance(in_dtype, (types.Float, types.NPDatetime, types.NPTimedelta))


def is_nullable(typ):
    return bodo.utils.utils.is_array_typ(typ, False) and (
        not isinstance(typ, types.Array) or is_dtype_nullable(typ.dtype)
    )


class BodoError(NumbaError):
    """Bodo error that is a regular exception to allow typing pass to catch it.
    Numba will handle it in a special way to remove any context information
    when printing so that it only prints the error message and code location.
    """

    def __init__(self, msg, loc=None, locs_in_msg=None):
        if locs_in_msg is None:
            self.locs_in_msg = []
        else:
            self.locs_in_msg = locs_in_msg
        highlight = numba.core.errors.termcolor().errmsg
        super(BodoError, self).__init__(highlight(msg), loc)


class BodoException(Exception):
    """Bodo exception that inherits from Exception to allow typing pass to catch it
    and potentially transform the IR.
    """


class BodoNotConstError(Exception):
    """Indicates that a constant value is expected but non-constant is provided.
    Only used in partial typing pass to enable IR transformation. Inherits from regular
    Exception class for this purpose.
    """


class BodoConstUpdatedError(Exception):
    """Indicates that a constant value is expected but the input list/dict/set is
    updated in place. Only used in partial typing pass to enable error checking.
    """


def raise_const_error(msg):
    """raises an error indicating that a constant value is expected with the given
    message 'msg'.
    Raises BodoNotConstError during partial type inference, and BodoError otherwise.
    """
    if bodo.transforms.typing_pass.in_partial_typing:
        bodo.transforms.typing_pass.typing_transform_required = True
        raise BodoNotConstError(msg)
    else:
        raise BodoError(msg)


def raise_bodo_error(msg, loc=None):
    """Raises BodoException during partial typing in case typing transforms can handle
    the issue. Otherwise, raises BodoError.
    """
    if bodo.transforms.typing_pass.in_partial_typing:
        bodo.transforms.typing_pass.typing_transform_required = True
        raise BodoException(msg)
    else:
        locs = [] if loc is None else [loc]
        raise BodoError(msg, locs_in_msg=locs)


class BodoWarning(Warning):
    """
    Warning class for Bodo-related potential issues such as prevention of
    parallelization by unsupported functions.
    """


def get_udf_error_msg(context_str, error):
    """Return error message for UDF-related errors. Adds location of UDF error
    to message.
    context_str: Context for UDF error, e.g. "Dataframe.apply()"
    error: UDF error
    """
    # the error could be a Numba TypingError with 'msg' and 'loc' attributes, or just
    # a regular Python Exception/Error with 'args' attribute
    msg = ""
    if hasattr(error, "msg"):
        msg = str(error.msg)
    if hasattr(error, "args") and error.args:
        # TODO(ehsan): can Exception have more than one arg?
        msg = str(error.args[0])

    loc = ""
    if hasattr(error, "loc") and error.loc is not None:
        loc = error.loc.strformat()

    return f"{context_str}: user-defined function not supported: {msg}\n{loc}"


class FileInfo:
    """This object is passed to ForceLiteralArg to convert argument
    to FilenameType instead of Literal"""

    def __init__(self):
        # if not None, it is a string that needs to be concatenated to input string in
        # get_schema() to get full path for retrieving schema
        self._concat_str = None
        # whether _concat_str should be concatenated on the left
        self._concat_left = None

    def get_schema(self, fname):
        """get dataset schema from file name"""
        full_path = self.get_full_filename(fname)
        return self._get_schema(full_path)

    def set_concat(self, concat_str, is_left):
        """set input string concatenation parameters"""
        self._concat_str = concat_str
        self._concat_left = is_left

    def _get_schema(self, fname):
        # should be implemented in subclasses
        raise NotImplementedError

    def get_full_filename(self, fname):
        """get full path with concatenation if necessary"""
        if self._concat_str is None:
            return fname

        if self._concat_left:
            return self._concat_str + fname

        return fname + self._concat_str


class FilenameType(types.StringLiteral):
    """Arguments of Bodo functions that are a constant literal are
    converted to this type instead of plain Literal to allow us
    to reuse the cache for differing file names that have the
    same schema. All FilenameType instances have the same hash
    to allow comparison of different instances. Equality is based
    on the schema (not the file name)."""

    def __init__(self, fname, finfo):
        self.fname = fname
        self.schema = finfo.get_schema(fname)
        super(FilenameType, self).__init__(self.fname)

    def __hash__(self):
        # fixed number to ensure every FilenameType hashes equally
        return 37

    def __eq__(self, other):
        if isinstance(other, types.FilenameType):
            assert self.schema is not None
            assert other.schema is not None
            return self.schema == other.schema
        else:
            return False


types.FilenameType = FilenameType

# Data model, unboxing and lower cast are the same as unicode to
# allow passing different file names to compiled code (note that if
# data model is literal the file name would be part of the binary code)

# datamodel
register_model(types.FilenameType)(numba.cpython.unicode.UnicodeModel)

# unbox
unbox(types.FilenameType)(numba.cpython.unicode.unbox_unicode_str)

# lower cast
@lower_cast(types.FilenameType, types.unicode_type)
def cast_filename_to_unicode(context, builder, fromty, toty, val):
    # do nothing
    return val


# sentinel value representing non-constant values
class NotConstant:
    pass


NOT_CONSTANT = NotConstant()


def is_overload_none(val):
    return val is None or val == types.none or getattr(val, "value", False) is None


def is_overload_constant_bool(val):
    return (
        isinstance(val, bool)
        or isinstance(val, types.BooleanLiteral)
        or ((isinstance(val, types.Omitted) and isinstance(val.value, bool)))
    )


def is_overload_bool(val):
    return isinstance(val, types.Boolean) or is_overload_constant_bool(val)


def is_overload_constant_str(val):
    return (
        isinstance(val, str)
        or (isinstance(val, types.StringLiteral) and isinstance(val.literal_value, str))
        or ((isinstance(val, types.Omitted) and isinstance(val.value, str)))
    )


def is_overload_constant_bytes(val):
    """Checks if the specified value is a binary constant"""
    return (
        isinstance(val, bytes)
        # Numba doesn't have a coresponding literal type for byte literals
        # or (isinstance(val, types.BinaryLiteral) and isinstance(val.literal_value, bytes))
        or ((isinstance(val, types.Omitted) and isinstance(val.value, bytes)))
    )


def is_overload_constant_list(val):
    """return True if 'val' is a constant list in overload. Currently considers tuples
    as well since tuples and lists are interchangable in most Pandas APIs
    (TODO: revisit).
    """
    return (
        isinstance(val, (list, tuple))
        or (isinstance(val, types.Omitted) and isinstance(val.value, tuple))
        or is_initial_value_list_type(val)
        or isinstance(val, types.LiteralList)
        or isinstance(val, bodo.utils.typing.ListLiteral)
        or (
            isinstance(val, types.BaseTuple)
            and all(is_literal_type(t) for t in val.types)
            # avoid const dict values stored as const tuple
            and (
                not val.types
                or val.types[0] != types.StringLiteral(CONST_DICT_SENTINEL)
            )
        )
    )


def is_overload_constant_tuple(val):
    return (
        isinstance(val, tuple)
        or (isinstance(val, types.Omitted) and isinstance(val.value, tuple))
        or (
            isinstance(val, types.BaseTuple)
            and all(get_overload_const(t) is not NOT_CONSTANT for t in val.types)
        )
    )


def is_initial_value_type(t):
    """return True if 't' is a dict/list container with initial constant values"""
    if not isinstance(t, types.InitialValue) or t.initial_value is None:
        return False
    vals = t.initial_value
    if isinstance(vals, dict):
        vals = vals.values()
    # Numba 0.51 assigns unkown or Poison to values sometimes
    # see test_groupby_agg_const_dict::impl16
    return not any(
        isinstance(v, (types.Poison, numba.core.interpreter._UNKNOWN_VALUE))
        for v in vals
    )


def is_initial_value_list_type(t):
    """return True if 't' is a list with initial constant values"""
    return isinstance(t, types.List) and is_initial_value_type(t)


def is_initial_value_dict_type(t):
    """return True if 't' is a dict with initial constant values"""
    return isinstance(t, types.DictType) and is_initial_value_type(t)


def is_overload_constant_dict(val):
    """const dict values are stored as a const tuple with a sentinel"""

    return (
        (
            isinstance(val, types.LiteralStrKeyDict)
            and all(is_literal_type(v) for v in val.types)
        )
        or is_initial_value_dict_type(val)
        or isinstance(val, DictLiteral)
        or (
            isinstance(val, types.BaseTuple)
            and val.types
            and val.types[0] == types.StringLiteral(CONST_DICT_SENTINEL)
        )
        or isinstance(val, dict)
    )


def is_overload_constant_number(val):
    return is_overload_constant_int(val) or is_overload_constant_float(val)


def is_overload_constant_nan(val):
    """Returns True if val is a constant np.nan. This is useful
    for situations where setting a null value may be allowed,
    but general float support would have a different implementation.
    """
    return is_overload_constant_float(val) and np.isnan(get_overload_const_float(val))


def is_overload_constant_float(val):
    return isinstance(val, float) or (
        (isinstance(val, types.Omitted) and isinstance(val.value, float))
    )


def is_overload_int(val):
    return is_overload_constant_int(val) or isinstance(val, types.Integer)


def is_overload_constant_int(val):
    return (
        isinstance(val, int)
        or (
            isinstance(val, types.IntegerLiteral) and isinstance(val.literal_value, int)
        )
        or ((isinstance(val, types.Omitted) and isinstance(val.value, int)))
    )


def is_overload_bool_list(val):
    """return True if 'val' is a constant list type with all constant boolean values"""
    return is_overload_constant_list(val) and all(
        is_overload_constant_bool(v) for v in get_overload_const_list(val)
    )


def is_overload_true(val):
    return (
        val == True
        or val == types.BooleanLiteral(True)
        or getattr(val, "value", False) is True
    )


def is_overload_false(val):
    return (
        val == False
        or val == types.BooleanLiteral(False)
        or getattr(val, "value", True) is False
    )


def is_overload_zero(val):
    return val == 0 or val == types.IntegerLiteral(0) or getattr(val, "value", -1) == 0


def is_overload_str(val, const):
    return (
        val == const
        or val == types.StringLiteral(const)
        or getattr(val, "value", -1) == const
    )


# TODO: refactor with get_literal_value()
def get_overload_const(val):
    """Get constant value for overload input. Returns NOT_CONSTANT if not constant.
    'val' can be a python value, an Omitted type, a literal type, or other Numba type
    (in case of non-constant).
    Supports None, bool, int, str, and tuple values.
    """
    from bodo.hiframes.datetime_timedelta_ext import _no_input

    # sometimes Dispatcher objects become TypeRef, see test_groupby_agg_const_dict
    if isinstance(val, types.TypeRef):
        val = val.instance_type
    if val == types.none:
        return None
    if val is _no_input:
        return _no_input
    # actual value
    if val is None or isinstance(val, (bool, int, float, str, tuple, types.Dispatcher)):
        return val
    # Omitted case
    if isinstance(val, types.Omitted):
        return val.value
    # Literal value
    # LiteralList needs special handling since it may store literal values instead of
    # actual constants, see test_groupby_dead_col_multifunc
    if isinstance(val, types.LiteralList):
        out_list = []
        for v in val.literal_value:
            const_val = get_overload_const(v)
            if const_val == NOT_CONSTANT:
                return NOT_CONSTANT
            else:
                out_list.append(const_val)
        return out_list
    if isinstance(val, types.Literal):
        return val.literal_value
    if isinstance(val, types.Dispatcher):
        return val
    if isinstance(val, types.BaseTuple):
        out_list = []
        for v in val.types:
            const_val = get_overload_const(v)
            if const_val == NOT_CONSTANT:
                return NOT_CONSTANT
            else:
                out_list.append(const_val)
        return tuple(out_list)
    if is_initial_value_list_type(val):
        return val.initial_value
    if is_literal_type(val):
        return get_literal_value(val)
    return NOT_CONSTANT


def element_type(val):
    """Return the element type of a scalar or array"""
    if isinstance(val, (types.List, types.ArrayCompatible)):
        if isinstance(val.dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
            return val.dtype.elem_type
        # Bytes type is array compatible, but should be treated as scalar
        if val == bodo.bytes_type:
            return bodo.bytes_type
        return val.dtype
    return types.unliteral(val)


def can_replace(to_replace, value):
    """Return whether value can replace to_replace"""
    return (
        is_common_scalar_dtype([to_replace, value])
        # Float cannot replace Integer
        and not (
            isinstance(to_replace, types.Integer) and isinstance(value, types.Float)
        )
        # Integer and Float cannot replace Boolean
        and not (
            isinstance(to_replace, types.Boolean)
            and isinstance(value, (types.Integer, types.Float))
        )
    )


# string representation of basic types for printing
_const_type_repr = {str: "string", bool: "boolean", int: "integer"}


def ensure_constant_arg(fname, arg_name, val, const_type):
    """Make sure argument 'val' to overload of function 'fname' is a constant of type
    'const_type'. Otherwise, raise BodoError.
    """
    const_val = get_overload_const(val)
    const_type_name = _const_type_repr.get(const_type, str(const_type))

    if not isinstance(const_val, const_type):
        raise BodoError(
            f"{fname}(): argument '{arg_name}' should be a constant "
            f"{const_type_name} not {val}"
        )


def ensure_constant_values(fname, arg_name, val, const_values):
    """Make sure argument 'val' to overload of function 'fname' is one of the values in
    'const_values'. Otherwise, raise BodoError.
    """
    const_val = get_overload_const(val)

    if const_val not in const_values:
        raise BodoError(
            f"{fname}(): argument '{arg_name}' should be a constant value in "
            f"{const_values} not '{const_val}'"
        )


def check_unsupported_args(
    fname,
    args_dict,
    arg_defaults_dict,
    package_name="pandas",
    fn_str=None,
    module_name="",
):
    """Check for unsupported arguments for function 'fname', and raise an error if any
    value other than the default is provided.
    'args_dict' is a dictionary of provided arguments in overload.
    'arg_defaults_dict' is a dictionary of default values for unsupported arguments.

    'package_name' is used to differentiate by various libraries in documentation links (i.e. numpy, pandas)

    'module_name' is used for libraries that are split into multiple different files per module.
    """
    from bodo.hiframes.datetime_timedelta_ext import _no_input

    assert len(args_dict) == len(arg_defaults_dict)
    if fn_str == None:
        fn_str = f"{fname}()"
    error_message = ""
    unsupported = False
    for a in args_dict:
        v1 = get_overload_const(args_dict[a])
        v2 = arg_defaults_dict[a]
        if (
            v1 is NOT_CONSTANT
            or (v1 is not None and v2 is None)
            or (v1 is None and v2 is not None)
            or v1 != v2
            or (v1 is not _no_input and v2 is _no_input)
            or (v1 is _no_input and v2 is not _no_input)
        ):
            error_message = f"{fn_str}: {a} parameter only supports default value {v2}"
            unsupported = True
            break

    if unsupported and package_name == "pandas":
        if module_name == "IO":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/io.html).\n"
        elif module_name == "General":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/general.html).\n"
        elif module_name == "DataFrame":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/dataframe.html).\n"
        elif module_name == "Window":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/window.html).\n"
        elif module_name == "GroupBy":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/groupby.html).\n"
        elif module_name == "Series":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/series.html).\n"
        elif module_name == "HeterogeneousSeries":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/series.html#heterogeneous-series).\n"
        elif module_name == "Index":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/indexapi.html).\n"
        elif module_name == "Timestamp":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/timestamp.html).\n"
        elif module_name == "Timedelta":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/timedelta.html).\n"
        elif module_name == "DateOffsets":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/pandas/dateoffsets.html).\n"

    elif unsupported and package_name == "ml":
        error_message += "\nPlease check supported ML operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/ml.html).\n"
    elif unsupported and package_name == "numpy":
        error_message += "\nPlease check supported Numpy operations here (https://docs.bodo.ai/latest/source/programming_with_bodo/bodo_api_reference/numpy.html).\n"
    if unsupported:
        raise BodoError(error_message)


def get_overload_const_tuple(val):
    if isinstance(val, tuple):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, tuple)
        return val.value
    if isinstance(val, types.BaseTuple):
        return tuple(get_overload_const(t) for t in val.types)


def get_overload_constant_dict(val):
    """get constant dict values from literal type (stored as const tuple)"""
    # LiteralStrKeyDict with all const values, e.g. {"A": ["B"]}
    # see test_groupby_agg_const_dict::impl4
    if isinstance(val, types.LiteralStrKeyDict):
        return {
            get_literal_value(k): get_literal_value(v)
            for k, v in val.literal_value.items()
        }
    if isinstance(val, DictLiteral):
        return val.literal_value
    if isinstance(val, dict):
        return val
    assert is_initial_value_dict_type(val) or (
        isinstance(val, types.BaseTuple)
        and val.types
        and val.types[0] == types.StringLiteral(CONST_DICT_SENTINEL)
    ), "invalid const dict"
    if isinstance(val, types.DictType):
        assert val.initial_value is not None, "invalid dict initial value"
        return val.initial_value

    # get values excluding sentinel
    items = [get_overload_const(v) for v in val.types[1:]]
    # create dict and return
    return {items[2 * i]: items[2 * i + 1] for i in range(len(items) // 2)}


def get_overload_const_str_len(val):
    if isinstance(val, str):
        return len(val)
    if isinstance(val, types.StringLiteral) and isinstance(val.literal_value, str):
        return len(val.literal_value)
    if isinstance(val, types.Omitted) and isinstance(val.value, str):
        return len(val.value)


def get_overload_const_list(val):
    """returns a constant list from type 'val', which could be a single value
    literal, a constant list or a constant tuple.
    """
    if isinstance(val, (list, tuple)):
        return val
    if isinstance(val, types.Omitted) and isinstance(val.value, tuple):
        return val.value
    if is_initial_value_list_type(val):
        return val.initial_value
    if isinstance(val, types.LiteralList):
        return [get_literal_value(v) for v in val.literal_value]
    if isinstance(val, bodo.utils.typing.ListLiteral):
        return val.literal_value
    if isinstance(val, types.Omitted):
        return [val.value]
    # literal case
    if isinstance(val, types.Literal):
        return [val.literal_value]
    if isinstance(val, types.BaseTuple) and all(is_literal_type(t) for t in val.types):
        return tuple(get_literal_value(t) for t in val.types)


def get_overload_const_str(val):
    if isinstance(val, str):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, str)
        return val.value
    # literal case
    if isinstance(val, types.StringLiteral):
        assert isinstance(val.literal_value, str)
        return val.literal_value
    raise BodoError("{} not constant string".format(val))


def get_overload_const_bytes(val):
    """Gets the bytes value from the possibly wraped value.
    Val must actually be a constant byte type, or this fn will throw an error
    """
    if isinstance(val, bytes):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, bytes)
        return val.value
    # Numba has no eqivalent literal type for bytes
    raise BodoError("{} not constant binary".format(val))


def get_overload_const_int(val):
    if isinstance(val, int):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, int)
        return val.value
    # literal case
    if isinstance(val, types.IntegerLiteral):
        assert isinstance(val.literal_value, int)
        return val.literal_value
    raise BodoError("{} not constant integer".format(val))


def get_overload_const_float(val):
    if isinstance(val, float):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, float)
        return val.value
    raise BodoError("{} not constant float".format(val))


def get_overload_const_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, bool)
        return val.value
    # literal case
    if isinstance(val, types.BooleanLiteral):
        assert isinstance(val.literal_value, bool)
        return val.literal_value
    raise BodoError("{} not constant boolean".format(val))


def is_const_func_type(t):
    """check if 't' is a constant function type"""
    return isinstance(
        t,
        (
            types.MakeFunctionLiteral,
            bodo.utils.typing.FunctionLiteral,
            types.Dispatcher,
        ),
    )


def get_overload_const_func(val, func_ir):
    """get constant function object or ir.Expr.make_function from function type"""
    if isinstance(val, (types.MakeFunctionLiteral, bodo.utils.typing.FunctionLiteral)):
        func = val.literal_value
        # Handle functions that are currently make_function expressions from BodoSQL
        if isinstance(func, ir.Expr) and func.op == "make_function":
            assert (
                func_ir is not None
            ), "Function expression is make_function but there is no existing IR"
            func = numba.core.ir_utils.convert_code_obj_to_function(func, func_ir)
        return func
    if isinstance(val, types.Dispatcher):
        return val.dispatcher.py_func
    if isinstance(val, CPUDispatcher):
        return val.py_func
    raise BodoError("'{}' not a constant function type".format(val))


def is_heterogeneous_tuple_type(t):
    """check if 't' is a heterogeneous tuple type (or similar, e.g. constant list)"""
    if is_overload_constant_list(t):
        # LiteralList values may be non-constant
        if isinstance(t, types.LiteralList):
            t = types.BaseTuple.from_types(t.types)
        else:
            t = bodo.typeof(tuple(get_overload_const_list(t)))

    if isinstance(t, bodo.NullableTupleType):
        t = t.tuple_typ

    return isinstance(t, types.BaseTuple) and not isinstance(t, types.UniTuple)


def parse_dtype(dtype, func_name=None):
    """Parse dtype type specified in various forms into actual numba type
    (e.g. StringLiteral("int32") to types.int32)
    """
    if isinstance(dtype, types.TypeRef):
        return dtype.instance_type

    # handle constructor functions, e.g. Series.astype(float)
    if isinstance(dtype, types.Function):
        # TODO: other constructor functions?
        if dtype.key[0] == float:
            dtype = types.StringLiteral("float")
        elif dtype.key[0] == int:
            dtype = types.StringLiteral("int")
        elif dtype.key[0] == bool:
            dtype = types.StringLiteral("bool")
        elif dtype.key[0] == str:
            dtype = bodo.string_type

    if isinstance(dtype, types.DTypeSpec):
        return dtype.dtype

    # input is array dtype already
    if isinstance(dtype, types.Number) or dtype == bodo.string_type:
        return dtype

    try:
        d_str = get_overload_const_str(dtype)
        if d_str.startswith("Int") or d_str.startswith("UInt"):
            return bodo.libs.int_arr_ext.typeof_pd_int_dtype(
                pd.api.types.pandas_dtype(d_str), None
            )
        if d_str == "boolean":
            return bodo.libs.bool_arr_ext.boolean_dtype
        return numba.np.numpy_support.from_dtype(np.dtype(d_str))
    except:
        pass
    if func_name is not None:
        raise BodoError(f"{func_name}(): invalid dtype {dtype}")
    else:
        raise BodoError(f"invalid dtype {dtype}")


def is_list_like_index_type(t):
    """Types that can be similar to list for indexing Arrays, Series, etc.
    Tuples are excluded due to indexing semantics.
    """
    from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.libs.bool_arr_ext import boolean_array

    # TODO: include datetimeindex/timedeltaindex?

    return (
        isinstance(t, types.List)
        or (isinstance(t, types.Array) and t.ndim == 1)
        or isinstance(t, (NumericIndexType, RangeIndexType))
        or isinstance(t, SeriesType)
        # or isinstance(t, IntegerArrayType)  # TODO: is this necessary?
        or t == boolean_array
    )


def is_tuple_like_type(t):
    """return True of 't' is a tuple-like type such as tuples or literal list that
    could be used in constant sized DataFrame, Series or Index.
    """
    return (
        isinstance(t, types.BaseTuple)
        or is_heterogeneous_tuple_type(t)
        or isinstance(t, bodo.hiframes.pd_index_ext.HeterogeneousIndexType)
    )


def get_index_names(t, func_name, default_name):
    """get name(s) of index type 't', assuming constant string literal name(s) are used.
    otherwise, throw error.
    """
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType

    err_msg = "{}: index name should be a constant string".format(func_name)

    # MultIndex has multiple names
    if isinstance(t, MultiIndexType):
        names = []
        for i, n_typ in enumerate(t.names_typ):
            if n_typ == types.none:
                names.append("level_{}".format(i))
                continue
            if not is_overload_constant_str(n_typ):
                raise BodoError(err_msg)
            names.append(get_overload_const_str(n_typ))
        return tuple(names)

    # other indices have a single name
    if t.name_typ == types.none:
        return (default_name,)
    if not is_overload_constant_str(t.name_typ):
        raise BodoError(err_msg)
    return (get_overload_const_str(t.name_typ),)


def get_index_data_arr_types(t):
    """get array type corresponding to Index type 't'"""
    from bodo.hiframes.pd_index_ext import (
        BinaryIndexType,
        CategoricalIndexType,
        DatetimeIndexType,
        IntervalIndexType,
        NumericIndexType,
        PeriodIndexType,
        RangeIndexType,
        StringIndexType,
        TimedeltaIndexType,
    )
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType

    if isinstance(t, MultiIndexType):
        return tuple(t.array_types)

    if isinstance(t, (RangeIndexType, PeriodIndexType)):
        return (types.Array(types.int64, 1, "C"),)

    if isinstance(
        t,
        (
            NumericIndexType,
            StringIndexType,
            BinaryIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            CategoricalIndexType,
            IntervalIndexType,
        ),
    ):
        return (t.data,)

    raise BodoError(f"Invalid index type {t}")


def get_index_type_from_dtype(t):
    """get Index type that can hold dtype 't' values."""
    from bodo.hiframes.pd_index_ext import (
        BinaryIndexType,
        CategoricalIndexType,
        DatetimeIndexType,
        NumericIndexType,
        StringIndexType,
        TimedeltaIndexType,
    )

    if t in [bodo.hiframes.pd_timestamp_ext.pd_timestamp_type, bodo.datetime64ns]:
        return DatetimeIndexType(types.none)

    if t in [
        bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type,
        bodo.timedelta64ns,
    ]:
        return TimedeltaIndexType(types.none)

    if t == bodo.string_type:
        return StringIndexType(types.none)

    if t == bodo.bytes_type:
        return BinaryIndexType(types.none)

    if isinstance(t, (types.Integer, types.Float, types.Boolean)):
        return NumericIndexType(t, types.none)

    if isinstance(t, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        return CategoricalIndexType(bodo.CategoricalArrayType(t))

    raise BodoError(f"Cannot convert dtype {t} to index type")


def get_val_type_maybe_str_literal(value):
    """Get type of value, using StringLiteral if possible"""
    t = numba.typeof(value)
    if isinstance(value, str):
        t = types.StringLiteral(value)
    return t


def get_index_name_types(t):
    """get name types of index type 't'. MultiIndex has multiple names but others have
    a single name.
    """
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType

    # MultIndex has multiple names
    if isinstance(t, MultiIndexType):
        return t.names_typ

    # other indices have a single name
    return (t.name_typ,)


# Check if boxing if defined for SliceLiteral. If not provide the implementation.
if types.SliceLiteral in numba.core.pythonapi._boxers.functions:
    warnings.warn("SliceLiteral boxing has been implemented in Numba")
else:

    @box(types.SliceLiteral)
    def box_slice_literal(typ, val, c):
        """box slice literal by constructing a slice from start, stop, and step"""
        slice_val = typ.literal_value
        slice_fields = []
        for field_name in ("start", "stop", "step"):
            field_obj = getattr(typ.literal_value, field_name)
            field_val = (
                c.pyapi.make_none()
                if field_obj is None
                else c.pyapi.from_native_value(
                    types.literal(field_obj), field_obj, c.env_manager
                )
            )
            slice_fields.append(field_val)
        # TODO: Replace with a CPython API once added to pythonapi.py
        slice_call_obj = c.pyapi.unserialize(c.pyapi.serialize_object(slice))
        slice_obj = c.pyapi.call_function_objargs(slice_call_obj, slice_fields)
        # Decref objects
        for a in slice_fields:
            c.pyapi.decref(a)
        c.pyapi.decref(slice_call_obj)
        return slice_obj


class ListLiteral(types.Literal):
    """class for literal lists, only used when Bodo forces an argument to be a literal
    list (e.g. in typing pass for groupby/join/sort_values).
    """


types.Literal.ctor_map[list] = ListLiteral
register_model(ListLiteral)(models.OpaqueModel)


@unbox(ListLiteral)
def unbox_list_literal(typ, obj, c):
    # A list literal is a dummy value
    return NativeValue(c.context.get_dummy_value())


@box(ListLiteral)
def box_list_literal(typ, val, c):
    """box list literal by boxing individual elements and packing them into a list obj"""
    list_val = typ.literal_value
    item_objs = [
        c.pyapi.from_native_value(types.literal(v), v, c.env_manager) for v in list_val
    ]
    out_list_obj = c.pyapi.list_pack(item_objs)
    for a in item_objs:
        c.pyapi.decref(a)
    return out_list_obj


@lower_cast(ListLiteral, types.List)
def list_literal_to_list(context, builder, fromty, toty, val):
    """cast literal list to regular list to support operations like iter()"""
    # lower a const tuple and convert to list inside the function to avoid Numba errors
    list_vals = tuple(fromty.literal_value)
    # remove 'reflected' from list type to avoid errors
    res_type = types.List(toty.dtype)
    return context.compile_internal(
        builder,
        lambda: list(list_vals),
        res_type(),
        [],
    )  # pragma: no cover


# TODO(ehsan): allow modifying the value similar to initial value containers?
class DictLiteral(types.Literal):
    """class for literal dictionaries, only used when Bodo forces an argument to be a
    literal dict (e.g. in typing pass for dataframe/groupby/join/sort_values).
    """


types.Literal.ctor_map[dict] = DictLiteral
register_model(DictLiteral)(models.OpaqueModel)


@unbox(DictLiteral)
def unbox_dict_literal(typ, obj, c):
    # A dict literal is a dummy value
    return NativeValue(c.context.get_dummy_value())


# literal type for functions (to handle function arguments to map/apply methods)
# similar to MakeFunctionLiteral
class FunctionLiteral(types.Literal, types.Opaque):
    """Literal type for function objects (i.e. pytypes.FunctionType)"""


types.Literal.ctor_map[pytypes.FunctionType] = FunctionLiteral
register_model(FunctionLiteral)(models.OpaqueModel)


# dummy unbox to avoid errors when function is passed as argument
@unbox(FunctionLiteral)
def unbox_func_literal(typ, obj, c):
    return NativeValue(obj)


# groupby.agg() can take a constant dictionary with a UDF in values. Typer of Numba's
# typed.Dict trys to get the type of the UDF value, which is not possible. This hack
# makes a dummy type available to Numba so that type inference works.
types.MakeFunctionLiteral._literal_type_cache = types.MakeFunctionLiteral(lambda: 0)


# type used to pass metadata to type inference functions
# see untyped_pass.py and df.pivot_table()
class MetaType(types.Type):
    def __init__(self, meta):
        self.meta = meta
        super(MetaType, self).__init__("MetaType({})".format(meta))

    def can_convert_from(self, typingctx, other):
        return True

    @property
    def key(self):
        # XXX this is needed for _TypeMetaclass._intern to return the proper
        # cached instance in case meta is changed
        # (e.g. TestGroupBy -k pivot -k cross)
        return tuple(self.meta)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


register_model(MetaType)(models.OpaqueModel)


def is_literal_type(t):
    """return True if 't' represents a data type with known compile-time constant value"""
    # sometimes Dispatcher objects become TypeRef, see test_groupby_agg_const_dict
    if isinstance(t, types.TypeRef):
        t = t.instance_type
    return (
        # LiteralStrKeyDict is not always a literal since its values are not necessarily
        # constant
        (
            isinstance(t, (types.Literal, types.Omitted))
            and not isinstance(t, types.LiteralStrKeyDict)
        )
        or t == types.none  # None type is always literal since single value
        or isinstance(t, types.Dispatcher)
        # LiteralStrKeyDict is a BaseTuple in Numba 0.51 also
        or (isinstance(t, types.BaseTuple) and all(is_literal_type(v) for v in t.types))
        # List/Dict types preserve const initial values in Numba 0.51
        or is_initial_value_type(t)
        # dtype literals should be treated as literals
        or isinstance(t, (types.DTypeSpec, types.Function))
        or isinstance(t, bodo.libs.int_arr_ext.IntDtype)
        or t
        in (bodo.libs.bool_arr_ext.boolean_dtype, bodo.libs.str_arr_ext.string_dtype)
        # values like np.sum could be passed as UDFs and are technically literals
        # See test_groupby_agg_func_udf
        or isinstance(t, types.Function)
        # Index with known values
        or is_overload_constant_index(t)
        # Series with known values
        or is_overload_constant_series(t)
        or is_overload_constant_dict(t)
    )


def is_overload_constant_index(t):
    """return True if 't' is a Index data type with known compile time values"""
    from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

    return (
        isinstance(t, HeterogeneousIndexType)
        and is_literal_type(t.data)
        and is_literal_type(t.name_type)
    )


def get_overload_constant_index(t):
    """return compile time constant value for Index type 't' (assuming it is a literal)"""
    assert is_overload_constant_index(t)
    return pd.Index(get_literal_value(t.data), name=get_literal_value(t.name_type))


def is_overload_constant_series(t):
    """return True if 't' is a Series data type with known compile time values"""
    from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType

    return (
        isinstance(t, (SeriesType, HeterogeneousSeriesType))
        and is_literal_type(t.data)
        and is_literal_type(t.index)
        and is_literal_type(t.name_typ)
    )


def get_overload_constant_series(t):
    """return compile time constant value for Series type 't' (assuming it is a literal)"""
    assert is_overload_constant_series(t)
    return pd.Series(
        get_literal_value(t.data),
        get_literal_value(t.index),
        name=get_literal_value(t.name_typ),
    )


def get_literal_value(t):
    """return compile time constant value for type 't' (assuming it is a literal)"""
    # sometimes Dispatcher objects become TypeRef, see test_groupby_agg_const_dict
    if isinstance(t, types.TypeRef):
        t = t.instance_type
    assert is_literal_type(t)
    if t == types.none:
        return None
    if isinstance(t, types.Literal):
        # LiteralStrKeyDict with all const values, e.g. {"A": ["B"]}
        if isinstance(t, types.LiteralStrKeyDict):
            return {
                get_literal_value(k): get_literal_value(v)
                for k, v in t.literal_value.items()
            }
        # types.LiteralList stores values as Literal types so needs get_literal_value
        if isinstance(t, types.LiteralList):
            return [get_literal_value(v) for v in t.literal_value]
        return t.literal_value
    if isinstance(t, types.Omitted):
        return t.value
    if isinstance(t, types.BaseTuple):
        return tuple(get_literal_value(v) for v in t.types)
    if isinstance(t, types.Dispatcher):
        return t
    if is_initial_value_type(t):
        return t.initial_value
    if isinstance(t, (types.DTypeSpec, types.Function)):
        return t
    if isinstance(t, bodo.libs.int_arr_ext.IntDtype):
        return getattr(pd, str(t)[:-2])()
    if t == bodo.libs.bool_arr_ext.boolean_dtype:
        return pd.BooleanDtype()
    if t == bodo.libs.str_arr_ext.string_dtype:
        return pd.StringDtype()
    if is_overload_constant_index(t):
        return get_overload_constant_index(t)
    if is_overload_constant_series(t):
        return get_overload_constant_series(t)
    if is_overload_constant_dict(t):
        return get_overload_constant_dict(t)


def can_literalize_type(t, pyobject_to_literal=False):
    """return True if type 't' can have literal values"""
    return (
        t in (bodo.string_type, types.bool_)
        or isinstance(t, (types.Integer, types.List, types.SliceType, types.DictType))
        or (pyobject_to_literal and t == types.pyobject)
    )


def dtype_to_array_type(dtype):
    """get default array type for scalar dtype"""
    dtype = types.unliteral(dtype)

    # UDFs may return lists, but we store array of array for output
    if isinstance(dtype, types.List):
        dtype = dtype_to_array_type(dtype.dtype)

    convert_nullable = False

    # UDFs may use Optional types for setting array values.
    # These should use the nullable type of the non-null case
    if isinstance(dtype, types.Optional):
        dtype = dtype.type
        convert_nullable = True

    # string array
    if dtype == bodo.string_type:
        return bodo.string_array_type

    # binary array
    if dtype == bodo.bytes_type:
        return bodo.binary_array_type

    if bodo.utils.utils.is_array_typ(dtype, False):
        return bodo.ArrayItemArrayType(dtype)

    # categorical
    if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        return bodo.CategoricalArrayType(dtype)

    if isinstance(dtype, bodo.libs.int_arr_ext.IntDtype):
        return bodo.IntegerArrayType(dtype.dtype)

    if dtype == types.bool_:
        return bodo.boolean_array

    if dtype == bodo.datetime_date_type:
        return bodo.hiframes.datetime_date_ext.datetime_date_array_type

    if isinstance(dtype, bodo.Decimal128Type):
        return bodo.DecimalArrayType(dtype.precision, dtype.scale)

    # struct array
    if isinstance(dtype, bodo.libs.struct_arr_ext.StructType):
        return bodo.StructArrayType(
            tuple(dtype_to_array_type(t) for t in dtype.data), dtype.names
        )

    # tuple array
    if isinstance(dtype, types.BaseTuple):
        return bodo.TupleArrayType(tuple(dtype_to_array_type(t) for t in dtype.types))

    # map array
    if isinstance(dtype, types.DictType):
        return bodo.MapArrayType(
            dtype_to_array_type(dtype.key_type),
            dtype_to_array_type(dtype.value_type),
        )

    # Timestamp/datetime are stored as dt64 array
    if dtype in (
        bodo.pd_timestamp_type,
        bodo.hiframes.datetime_datetime_ext.datetime_datetime_type,
    ):
        return types.Array(bodo.datetime64ns, 1, "C")

    # pd.Timedelta/datetime.timedelta values are stored as td64 arrays
    if dtype in (
        bodo.pd_timedelta_type,
        bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type,
    ):
        return types.Array(bodo.timedelta64ns, 1, "C")

    # regular numpy array
    if isinstance(
        dtype, (types.Number, types.Boolean, types.NPDatetime, types.NPTimedelta)
    ):
        arr = types.Array(dtype, 1, "C")
        # If this comes from an optional type try converting to
        # nullable.
        if convert_nullable:
            return to_nullable_type(arr)
        return arr

    raise BodoError(f"dtype {dtype} cannot be stored in arrays")  # pragma: no cover


def get_udf_out_arr_type(f_return_type, return_nullable=False):
    """get output array type of a UDF call, give UDF's scalar output type.
    E.g. S.map(lambda a: 2) -> array(int64)
    """

    # UDF output can be Optional if None is returned in a code path
    if isinstance(f_return_type, types.Optional):
        f_return_type = f_return_type.type
        return_nullable = True

    # unbox Timestamp to dt64 in Series
    if f_return_type == bodo.hiframes.pd_timestamp_ext.pd_timestamp_type:
        f_return_type = types.NPDatetime("ns")

    # unbox Timedelta to timedelta64 in Series
    if f_return_type == bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type:
        f_return_type = types.NPTimedelta("ns")

    out_arr_type = dtype_to_array_type(f_return_type)
    out_arr_type = to_nullable_type(out_arr_type) if return_nullable else out_arr_type
    return out_arr_type


def equality_always_false(t1, t2):
    """Helper function returns True if equality
    may exist between t1 and t2, but if so it will
    always return False.
    """
    # TODO: Enumerate all possible cases
    string_types = (
        types.UnicodeType,
        types.StringLiteral,
        types.UnicodeCharSeq,
    )
    return (isinstance(t1, string_types) and not isinstance(t2, string_types)) or (
        isinstance(t2, string_types) and not isinstance(t1, string_types)
    )


def types_equality_exists(t1, t2):
    """Determines if operator.eq is implemented between types
    t1 and t2. For efficient compilation time, you should first
    check if types are equal directly before calling this function.
    """
    typing_context = numba.core.registry.cpu_target.typing_context
    try:
        # Check if there is a valid equality between Series_type and
        # to_replace_type. If there isn't, we return a copy because we
        # know it is a no-op.
        typing_context.resolve_function_type(operator.eq, (t1, t2), {})
        return True
    except:
        return False


def is_hashable_type(t):
    """
    Determines if hash is implemented for type t.
    """
    # Use a whitelist of known hashable types to optimize
    # compilation time
    # TODO Enumerate all possible cases
    whitelist_types = (
        types.UnicodeType,
        types.StringLiteral,
        types.UnicodeCharSeq,
        types.Number,
    )
    whitelist_instances = (
        types.bool_,
        bodo.datetime64ns,
        bodo.timedelta64ns,
        bodo.pd_timestamp_type,
        bodo.pd_timedelta_type,
    )

    if isinstance(t, whitelist_types) or (t in whitelist_instances):
        return True

    typing_context = numba.core.registry.cpu_target.typing_context
    try:
        typing_context.resolve_function_type(hash, (t,), {})
        return True
    except:  # pragma: no cover
        return False


def to_nullable_type(t):
    """Converts types that cannot hold NAs to corresponding nullable types.
    For example, boolean_array is returned for Numpy array(bool_) and IntegerArray is
    returned for Numpy array(int).
    Converts data in DataFrame and Series types as well.
    """
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.hiframes.pd_series_ext import SeriesType

    if isinstance(t, DataFrameType):
        new_data = tuple(to_nullable_type(t) for t in t.data)
        return DataFrameType(new_data, t.index, t.columns, t.dist, t.is_table_format)

    if isinstance(t, SeriesType):
        return SeriesType(t.dtype, to_nullable_type(t.data), t.index, t.name_typ)

    if isinstance(t, types.Array):
        if t.dtype == types.bool_:
            return bodo.libs.bool_arr_ext.boolean_array

        if isinstance(t.dtype, types.Integer):
            return bodo.libs.int_arr_ext.IntegerArrayType(t.dtype)

    return t


def is_nullable_type(t):
    """return True if 't' is a nullable array type"""
    return t == to_nullable_type(t)


def is_iterable_type(t):
    """return True if 't' is an iterable type like list, array, Series, ..."""
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.hiframes.pd_series_ext import SeriesType

    return (
        bodo.utils.utils.is_array_typ(t, False)
        or isinstance(
            t,
            (SeriesType, DataFrameType, types.List, types.BaseTuple, types.LiteralList),
        )
        or bodo.hiframes.pd_index_ext.is_pd_index_type(t)
    )


def is_scalar_type(t):
    """
    returns True if 't' is a scalar type like integer, boolean, string, ...
    """
    return isinstance(t, (types.Boolean, types.Number, types.StringLiteral,)) or t in (
        bodo.datetime64ns,
        bodo.timedelta64ns,
        bodo.string_type,
        bodo.bytes_type,
        bodo.datetime_date_type,
        bodo.datetime_datetime_type,
        bodo.datetime_timedelta_type,
        bodo.pd_timestamp_type,
        bodo.pd_timedelta_type,
        bodo.month_end_type,
        bodo.week_type,
        bodo.date_offset_type,
        types.none,
    )


def is_common_scalar_dtype(scalar_types):
    """Returns True if a list of scalar types share a common
    Numpy type or are equal.
    """
    (_, found_common_typ) = get_common_scalar_dtype(scalar_types)
    return found_common_typ


def get_common_scalar_dtype(scalar_types):
    """
    Attempts to unify the list of passed in dtypes. Returns the tuple (common_type, True) on succsess,
    and (None, False) on failure."""
    scalar_types = [types.unliteral(a) for a in scalar_types]

    if len(scalar_types) == 0:
        raise_bodo_error(
            "Internal error, length of argument passed to get_common_scalar_dtype scalar_types is 0"
        )
    try:
        common_dtype = np.find_common_type(
            [numba.np.numpy_support.as_dtype(t) for t in scalar_types], []
        )
        # If we get an object dtype we do not have a common type.
        # Otherwise, the types can be used together
        if common_dtype != object:
            return (numba.np.numpy_support.from_dtype(common_dtype), True)

    # If we have a Bodo or Numba type that isn't implemented in
    # Numpy, we will get a NotImplementedError
    except NotImplementedError:
        pass

    # Timestamp/dt64 can be used interchangably
    # TODO: Should datetime.datetime also be included?
    if scalar_types[0] in (bodo.datetime64ns, bodo.pd_timestamp_type):
        for typ in scalar_types[1:]:
            if typ not in (bodo.datetime64ns, bodo.pd_timestamp_type):
                return (None, False)
        return (bodo.datetime64ns, True)

    # Timedelta/td64 can be used interchangably
    # TODO: Should datetime.timedelta also be included?
    if scalar_types[0] in (bodo.timedelta64ns, bodo.pd_timedelta_type):
        for typ in scalar_types[1:]:
            if scalar_types[0] not in (bodo.timedelta64ns, bodo.pd_timedelta_type):
                return (None, False)
        return (bodo.timedelta64ns, True)

    # If we don't have a common type, then all types need to be equal.
    # See: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    grouped_types = itertools.groupby(scalar_types)
    if next(grouped_types, True) and not next(grouped_types, False):
        return (scalar_types[0], True)

    return (None, False)


def find_common_np_dtype(arr_types):
    """finds common numpy dtype of array types using np.find_common_type"""
    return numba.np.numpy_support.from_dtype(
        np.find_common_type(
            [numba.np.numpy_support.as_dtype(t.dtype) for t in arr_types], []
        )
    )


def is_immutable_array(typ):
    """
    Returns if typ is an immutable array types. This is used for setitem
    error checking.
    """
    return isinstance(
        typ,
        (
            bodo.ArrayItemArrayType,
            bodo.MapArrayType,
        ),
    )


def get_nullable_and_non_nullable_types(array_of_types):
    """For each (non-)nullable type in the input list, add the corresponding (non)-nullable
    types to the list and return it. This makes checks for types more robust,
    specifically in pd.DataFrame.select_dtypes func."""

    all_types = []
    for typ in array_of_types:
        if typ == bodo.libs.bool_arr_ext.boolean_array:
            all_types.append(types.Array(types.bool_, 1, "C"))

        elif isinstance(typ, bodo.libs.int_arr_ext.IntegerArrayType):
            all_types.append(types.Array(typ.dtype, 1, "C"))

        elif isinstance(typ, types.Array):
            if typ.dtype == types.bool_:
                all_types.append(bodo.libs.bool_arr_ext.boolean_array)

            if isinstance(typ.dtype, types.Integer):
                all_types.append(bodo.libs.int_arr_ext.IntegerArrayType(typ.dtype))

        all_types.append(typ)

    return all_types


def _gen_objmode_overload(func, output_type, method_name=None, single_rank=False):
    """code gen for gen_objmode_func_overload and gen_objmode_method_overload"""
    func_spec = getfullargspec(func)

    assert func_spec.varargs is None, "varargs not supported"
    assert func_spec.varkw is None, "varkw not supported"

    defaults = [] if func_spec.defaults is None else func_spec.defaults
    n_pos_args = len(func_spec.args) - len(defaults)

    # Matplotlib specifies some arguments as `<deprecated parameter>`.
    # We can't support them, and it breaks our infrastructure, so omit them.

    args = func_spec.args[1:] if method_name else func_spec.args[:]
    arg_strs = []
    for i, arg in enumerate(func_spec.args):
        if i < n_pos_args:
            arg_strs.append(arg)
        elif str(defaults[i - n_pos_args]) != "<deprecated parameter>":
            arg_strs.append(arg + "=" + str(defaults[i - n_pos_args]))
        else:
            args.remove(arg)

    # Handle kwonly args. This assumes they have default values.
    if func_spec.kwonlyargs is not None:
        for arg in func_spec.kwonlyargs:
            # write args as arg=arg to handle kwonly requirement
            args.append(f"{arg}={arg}")
            arg_strs.append(f"{arg}={str(func_spec.kwonlydefaults[arg])}")

    sig = ", ".join(arg_strs)
    args = ", ".join(args)

    # workaround objmode string type name requirement by adding the type to types module
    # TODO: fix Numba's object mode to take type refs
    type_name = str(output_type)
    if not hasattr(types, type_name):
        type_name = f"objmode_type{ir_utils.next_label()}"
        setattr(types, type_name, output_type)

    if not method_name:
        # This Python function is going to be set at the global scope of this
        # module (bodo.utils.typing) so we need a name that won't clash
        func_name = func.__module__.replace(".", "_") + "_" + func.__name__ + "_func"

    call_str = f"self.{method_name}" if method_name else f"{func_name}"
    func_text = f"def overload_impl({sig}):\n"
    func_text += f"    def impl({sig}):\n"
    if single_rank:
        func_text += f"        if bodo.get_rank() == 0:\n"
        extra_indent = "    "
    else:
        extra_indent = ""
    func_text += f"        {extra_indent}with numba.objmode(res='{type_name}'):\n"
    func_text += f"            {extra_indent}res = {call_str}({args})\n"
    func_text += f"        return res\n"
    func_text += f"    return impl\n"

    loc_vars = {}
    # XXX For some reason numba needs a reference to the module or caching
    # won't work (and seems related to objmode).
    glbls = globals()
    if not method_name:
        glbls[func_name] = func
    exec(func_text, glbls, loc_vars)
    overload_impl = loc_vars["overload_impl"]
    return overload_impl


def gen_objmode_func_overload(func, output_type=None, single_rank=False):
    """generate an objmode overload to support function 'func' with output type
    'output_type'
    """
    try:
        overload_impl = _gen_objmode_overload(
            func, output_type, single_rank=single_rank
        )
        overload(func, no_unliteral=True)(overload_impl)
    except Exception:
        # If the module has changed in a way we can't support (i.e. varargs in matplotlib),
        # then don't do the overload
        pass


def gen_objmode_method_overload(
    obj_type, method_name, method, output_type=None, single_rank=False
):
    """generate an objmode overload_method to support method 'method'
    (named 'method_name') with output type 'output_type'.
    """
    try:
        overload_impl = _gen_objmode_overload(
            method, output_type, method_name, single_rank
        )
        overload_method(obj_type, method_name, no_unliteral=True)(overload_impl)
    except Exception:
        # If the module has changed in a way we can't support (i.e. varargs in matplotlib),
        # then don't do the overload
        pass


@infer
class NumTypeStaticGetItem(AbstractTemplate):
    """typer for getitem on number types in JIT code
    e.g. bodo.int64[::1] -> array(int64, 1, "C")
    """

    key = "static_getitem"

    def generic(self, args, kws):
        val, idx = args
        if isinstance(idx, slice) and (
            isinstance(val, types.NumberClass)
            or (
                isinstance(val, types.TypeRef)
                and isinstance(val.instance_type, (types.NPDatetime, types.NPTimedelta))
            )
        ):
            return signature(types.TypeRef(val.instance_type[idx]), *args)


@lower_builtin("static_getitem", types.NumberClass, types.SliceLiteral)
def num_class_type_static_getitem(context, builder, sig, args):
    # types don't have runtime values
    return context.get_dummy_value()


# dummy empty itertools implementation to avoid typing errors for series str
# flatten case
@overload(itertools.chain, no_unliteral=True)
def chain_overload():
    return lambda: [0]


@register_jitable
def from_iterable_impl(A):  # pragma: no cover
    """Internal call to support itertools.chain.from_iterable().
    Untyped pass replaces itertools.chain.from_iterable() with this call since class
    methods are not supported in Numba's typing
    """
    return bodo.utils.conversion.flatten_array(bodo.utils.conversion.coerce_to_array(A))


@intrinsic
def unliteral_val(typingctx, val=None):
    """converts the type of value 'val' to nonliteral"""

    def codegen(context, builder, signature, args):
        return args[0]

    return types.unliteral(val)(val), codegen


def create_unsupported_overload(fname):
    """Create an overload for unsupported function 'fname' that raises BodoError"""

    def overload_f(*a, **kws):
        raise BodoError("{} not supported yet".format(fname))

    return overload_f


def is_numpy_ufunc(func):
    """
    Determine if 'func' is a numpy ufunc. This is code written like np.abs.
    """
    # If a func is types.Function and its typing_key is a np.ufunc,
    # then we are working with a ufunc
    return isinstance(func, types.Function) and isinstance(func.typing_key, np.ufunc)


def is_builtin_function(func):
    """
    Determine if func is a builtin function typed by numba
    https://docs.python.org/3/library/builtins.html
    """
    # A function is builtin if its a types.Function
    # and the typing key is a python builtin.
    return isinstance(func, types.Function) and isinstance(
        func.typing_key, pytypes.BuiltinFunctionType
    )


def get_builtin_function_name(func):
    """
    Given a builtin function, which is a types.Function,
    returns the name of the function
    """
    # If func is a builtin, its name is
    # found with func.typing_key.__name__
    return func.typing_key.__name__


def construct_pysig(arg_names, defaults):
    """generate pysignature object for templates"""
    func_text = f"def stub("
    for arg in arg_names:
        func_text += arg
        if arg in defaults:
            # TODO: expand to other arg types?
            if isinstance(defaults[arg], str):
                func_text += f"='{defaults[arg]}'"
            else:
                func_text += f"={defaults[arg]}"
        func_text += ", "
    func_text += "):\n"
    func_text += "    pass\n"
    loc_vars = {}
    # TODO: Will some default args need globals?
    exec(func_text, {}, loc_vars)
    stub = loc_vars["stub"]
    return numba.core.utils.pysignature(stub)


def fold_typing_args(
    func_name,
    args,
    kws,
    arg_names,
    defaults,
    unsupported_arg_names=(),
):
    """
    Function that performs argument folding during the typing stage.
    This function uses the args, kws, argument names, defaults, and list
    of unsupported argument names to fold the arguments and perform basic
    error checking. This function does not check that each argument has the
    correct type, but it will check that unsupported arguments match the default
    value.

    Returns the pysig and the folded arguments that will be used to generate
    a signature.
    """
    # Ensure kws is a dictionary
    kws = dict(kws)

    # Check the number of args
    max_args = len(arg_names)
    passed_args = len(args) + len(kws)
    if passed_args > max_args:
        max_args_plural = "argument" if max_args == 1 else "arguments"
        passed_args_plural = "was" if passed_args == 1 else "were"
        raise BodoError(
            f"{func_name}(): Too many arguments specified. Function takes {max_args} {max_args_plural}, but {passed_args} {passed_args_plural} provided."
        )
    # Generate the pysig
    pysig = bodo.utils.typing.construct_pysig(arg_names, defaults)

    try:
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
    except Exception as e:
        # Typing Errors don't necessary show up for users in nested functions.
        # Use raise_bodo_error instead (in case a transformation removes the error).
        raise_bodo_error(f"{func_name}(): {e}")

    # Check unsupported args if there are any
    if unsupported_arg_names:
        # Generate the dictionaries for checking unsupported args
        unsupported_args = {}
        arg_defaults = {}
        for i, arg_name in enumerate(arg_names):
            if arg_name in unsupported_arg_names:
                assert (
                    arg_name in defaults
                ), f"{func_name}(): '{arg_name}' is unsupported but no default is provided"
                unsupported_args[arg_name] = folded_args[i]
                arg_defaults[arg_name] = defaults[arg_name]

        # Check unsupported args
        check_unsupported_args(func_name, unsupported_args, arg_defaults)

    return pysig, folded_args


def _is_pandas_numeric_dtype(dtype):
    # Pandas considers bool numeric as well: core/internals/blocks
    return isinstance(dtype, types.Number) or dtype == types.bool_


def type_col_to_index(col_names):
    """
    Takes a tuple of column names and generates the necessary types
    that would be generated by df.columns.
    Should match output of code generated by `generate_col_to_index_func_text`.
    """
    if all(isinstance(a, str) for a in col_names):
        return bodo.StringIndexType(None)
    elif all(isinstance(a, bytes) for a in col_names):
        return bodo.BinaryIndexType(None)
    elif all(isinstance(a, (int, float)) for a in col_names):  # pragma: no cover
        # TODO(ehsan): test
        if any(isinstance(a, (float)) for a in col_names):
            return bodo.NumericIndexType(types.float64)
        else:
            return bodo.NumericIndexType(types.int64)
    else:
        return bodo.hiframes.pd_index_ext.HeterogeneousIndexType(col_names)


def index_typ_from_dtype_name(elem_dtype, name):
    """
    Given a dtype and a name (which is either None or a string),
    returns a matching index type.
    """
    index_class = type(get_index_type_from_dtype(elem_dtype))
    if name is None:
        name_typ = None
    else:
        name_typ = types.StringLiteral(name)
    if index_class == bodo.hiframes.pd_index_ext.NumericIndexType:
        # Numeric requires the size
        index_typ = index_class(elem_dtype, name_typ)
    elif index_class == bodo.hiframes.pd_index_ext.CategoricalIndexType:
        # Categorical requires the categorical array
        index_typ = index_class(bodo.CategoricalArrayType(elem_dtype), name_typ)
    else:
        index_typ = index_class(name_typ)
    return index_typ


def is_safe_arrow_cast(lhs_scalar_typ, rhs_scalar_typ):
    """
    Determine if two scalar types which return False from
    'is_common_scalar_dtype' can be safely cast in an arrow
    filter expression. This is a white list of casts that
    are manually supported.
    """
    # TODO: Support more types
    if lhs_scalar_typ == types.unicode_type:
        # Cast is supported between string and timestamp
        return rhs_scalar_typ in (bodo.datetime64ns, bodo.pd_timestamp_type)
    elif lhs_scalar_typ in (bodo.datetime64ns, bodo.pd_timestamp_type):
        # Cast is supported between timestamp and string
        return rhs_scalar_typ == types.unicode_type
    return False


def register_type(type_name, type_value):
    """register a data type to be used in objmode blocks"""
    # check input
    if not isinstance(type_name, str):
        raise BodoError(
            f"register_type(): type name should be a string, not {type(type_name)}"
        )

    if not isinstance(type_value, types.Type):
        raise BodoError(
            f"register_type(): type value should be a valid data type, not {type(type_value)}"
        )

    if hasattr(types, type_name):
        raise BodoError(f"register_type(): type name '{type_name}' already exists")

    # add the data type to the "types" module used by Numba for type resolution
    # TODO(ehsan): develop a better solution since this is a bit hacky
    setattr(types, type_name, type_value)


# boxing TypeRef is necessary for passing type to objmode calls
@box(types.TypeRef)
def box_typeref(typ, val, c):
    return c.pyapi.unserialize(c.pyapi.serialize_object(typ.instance_type))


def check_objmode_output_type(ret_tup, ret_type):
    """check output values of objmode blocks to make sure they match the user-specified
    return type.
    `ret_tup` is a tuple of Python values being returned from objmode
    `ret_type` is the corresponding Numba tuple type
    """
    return tuple(_check_objmode_type(v, t) for v, t in zip(ret_tup, ret_type.types))


def _is_equiv_array_type(A, B):
    """return True if A and B are equivalent array types and can be converted without
    errors.
    """
    from bodo.libs.map_arr_ext import MapArrayType
    from bodo.libs.struct_arr_ext import StructArrayType

    # bodo.typeof() assigns StructArrayType to array of dictionary input if possible but
    # the data may actually be MapArrayType. This converts StructArrayType to
    # MapArrayType if possible and necessary.
    # StructArrayType can be converted to MapArrayType if all data arrays have the same
    # type.
    return (
        isinstance(A, StructArrayType)
        and isinstance(B, MapArrayType)
        and set(A.data) == {B.value_arr_type}
        and B.key_arr_type.dtype == bodo.string_type
    ) or (
        # Numpy array types that can be converted safely
        # see https://github.com/numba/numba/blob/306060a2e1eec194fa46b13c99a01651d944d657/numba/core/types/npytypes.py#L483
        isinstance(A, types.Array)
        and isinstance(B, types.Array)
        and A.ndim == B.ndim
        and A.dtype == B.dtype
        and B.layout in ("A", A.layout)
        and (A.mutable or not B.mutable)
        and (A.aligned or not B.aligned)
    )


def _fix_objmode_df_type(val, val_typ, typ):
    """fix output df of objmode to match user-specified type if possible"""
    from bodo.hiframes.pd_index_ext import RangeIndexType

    # distribution is just a hint and value can be cast trivially
    if val_typ.dist != typ.dist:
        val_typ = val_typ.copy(dist=typ.dist)

    # many users typically don't care about specifying the Index which defaults to
    # RangeIndex. We drop the Index and raise a warning to handle the common case.
    if isinstance(typ.index, RangeIndexType) and not isinstance(
        val_typ.index, RangeIndexType
    ):
        warnings.warn(
            BodoWarning(
                f"Dropping Index of objmode output dataframe since RangeIndexType specified in type annotation ({val_typ.index} to {typ.index})"
            )
        )
        val.reset_index(drop=True, inplace=True)
        val_typ = val_typ.copy(index=typ.index)

    # the user may not specify Index name type since it's usually not important
    if val_typ.index.name_typ != types.none and typ.index.name_typ == types.none:
        warnings.warn(
            BodoWarning(
                f"Dropping name field in Index of objmode output dataframe since none specified in type annotation ({val_typ.index} to {typ.index})"
            )
        )
        val_typ = val_typ.copy(index=typ.index)
        val.index.name = None

    # handle equivalent columns array types
    for i, (A, B) in enumerate(zip(val_typ.data, typ.data)):
        if _is_equiv_array_type(A, B):
            val_typ = val_typ.replace_col_type(val_typ.columns[i], B)

    # the user may not specify table format
    # NOTE: this will be unnecessary when table format is the default
    if val_typ.is_table_format and not typ.is_table_format:
        val_typ = val_typ.copy(is_table_format=False)

    # reorder df columns if possible to match the user-specified type
    if val_typ != typ:
        # sort column orders based on column names to see if the types can match
        val_cols = pd.Index(val_typ.columns)
        typ_cols = pd.Index(typ.columns)
        val_argsort = val_cols.argsort()
        typ_argsort = typ_cols.argsort()
        val_typ_sorted = val_typ.copy(
            data=tuple(np.array(val_typ.data)[val_argsort]),
            columns=tuple(val_cols[val_argsort]),
        )
        typ_sorted = typ.copy(
            data=tuple(np.array(typ.data)[typ_argsort]),
            columns=tuple(typ_cols[typ_argsort]),
        )
        if val_typ_sorted == typ_sorted:
            val_typ = typ
            val = val.reindex(columns=typ.columns)

    return val, val_typ


def _check_objmode_type(val, typ):
    """make sure the type of Python value `val` matches Numba type `typ`."""
    from bodo.hiframes.pd_dataframe_ext import DataFrameType

    val_typ = bodo.typeof(val)

    # handle dataframe type differences if possible
    if isinstance(typ, DataFrameType) and isinstance(val_typ, DataFrameType):
        val, val_typ = _fix_objmode_df_type(val, val_typ, typ)

    # some array types may be equivalent
    if _is_equiv_array_type(val_typ, typ):
        val_typ = typ

    # list/set reflection is irrelevant in objmode
    if isinstance(val_typ, (types.List, types.Set)):
        val_typ = val_typ.copy(reflected=False)

    # Numba casts number types liberally
    if isinstance(val_typ, (types.Integer, types.Float)) and isinstance(
        typ, (types.Integer, types.Float)
    ):
        return val

    if val_typ != typ:
        raise BodoError(
            f"Invalid objmode data type specified.\nUser specified:\t{typ}\nValue type:\t{val_typ}"
        )

    return val
