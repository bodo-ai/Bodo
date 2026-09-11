"""
Helper functions to enable typing.
"""

from __future__ import annotations

import copy
import itertools
import operator
import types as pytypes
import typing
import typing as pt
import warnings
from inspect import getfullargspec
from typing import Any

import numba
import numba.cpython.unicode
import numba.types
import numpy as np
import pandas as pd
import pyarrow as pa
from numba.core import cgutils, ir, ir_utils, types
from numba.core.errors import NumbaError
from numba.core.imputils import RefType, iternext_impl
from numba.core.registry import CPUDispatcher
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.core.utils import PYVERSION
from numba.extending import (
    NativeValue,
    box,
    infer,
    intrinsic,
    lower_builtin,
    lower_cast,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    unbox,
)

import bodo
from bodo import BodoWarning

# sentinel string used in typing pass that specifies a const tuple as a const dict.
# const tuple is used since there is no literal type for dict
CONST_DICT_SENTINEL = "$_bodo_const_dict_$"

# sentinel string used to indicate a dataframe Index (usually where column names used)
INDEX_SENTINEL = "$_bodo_index_"


list_cumulative = {"cumsum", "cumprod", "cummin", "cummax"}
Index = list[str | dict]
FileSchema: pt.TypeAlias = tuple[
    list[str], list, Index, list[int], list, list, list, pa.Schema
]


def is_timedelta_type(in_type):
    return (
        in_type
        in [
            bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type,
            bodo.hiframes.datetime_date_ext.datetime_timedelta_type,
        ]
        or in_type == bodo.types.timedelta64ns
    )


def is_dtype_nullable(in_dtype):
    """checks whether 'in_dtype' has sentinel NA values (as opposed to bitmap)"""
    return isinstance(in_dtype, (types.Float, types.NPDatetime, types.NPTimedelta))


def is_nullable(typ):
    return bodo.utils.utils.is_array_typ(typ, False) and (
        not isinstance(typ, types.Array) or is_dtype_nullable(typ.dtype)
    )


def is_nullable_ignore_sentinels(typ) -> bool:
    return bodo.utils.utils.is_array_typ(typ, False) and (
        not isinstance(typ, types.Array)
    )


def is_str_arr_type(t):
    """check if 't' is a regular or dictionary-encoded string array type
    TODO(ehsan): add other string types like np str array when properly supported
    """
    return t == bodo.types.string_array_type or t == bodo.types.dict_str_arr_type


def is_bin_arr_type(t):
    """check if 't' is a binary array type"""
    return t == bodo.types.binary_array_type


def type_has_unknown_cats(typ):
    """Return True if typ is a (or in case of tables has a) CategoricalArrayType with
    unknown categories (i.e. categories are created during runtime)

    Args:
        arr_type (types.Type): input array or table type

    Returns:
        bool: True if is/has categorical with unknown categories
    """
    return (
        isinstance(typ, bodo.types.CategoricalArrayType)
        and typ.dtype.categories is None
    ) or (
        isinstance(typ, bodo.types.TableType)
        and any(type_has_unknown_cats(t) for t in typ.type_to_blk.keys())
    )


def unwrap_typeref(typ: types.Type | types.TypeRef) -> types.Type:
    """return instance type if 'typ' is a TypeRef

    Args:
        typ (types.Type | types.TypeRef): input type

    Returns:
        types.Type: type without TypeRef
    """
    return typ.instance_type if isinstance(typ, types.TypeRef) else typ


def decode_if_dict_array(A):  # pragma: no cover
    return A


@overload(decode_if_dict_array)
def decode_if_dict_array_overload(A):
    """decodes input array if it is a dictionary-encoded array.
    Used as a fallback when dict array is not supported yet.
    """

    if isinstance(A, types.BaseTuple):
        n = len(A.types)
        func_text = "def bodo_decode_if_dict_array_basetuple(A):\n"
        res = ",".join(f"decode_if_dict_array(A[{i}])" for i in range(n))
        func_text += "  return ({}{})\n".format(res, "," if n == 1 else "")
        return bodo.utils.utils.bodo_exec(
            func_text, {"decode_if_dict_array": decode_if_dict_array}, {}, __name__
        )

    if isinstance(A, types.List):

        def bodo_decode_if_dict_array_list(A):  # pragma: no cover
            n = 0
            for a in A:
                n += 1
            ans = []
            for i in range(n):
                ans.append(decode_if_dict_array(A[i]))
            return ans

        return bodo_decode_if_dict_array_list
    if A == bodo.types.dict_str_arr_type:
        return lambda A: A._decode()  # pragma: no cover

    if isinstance(A, bodo.types.SeriesType):

        def bodo_decode_if_dict_array_series(A):  # pragma: no cover
            arr = bodo.hiframes.pd_series_ext.get_series_data(A)
            index = bodo.hiframes.pd_series_ext.get_series_index(A)
            name = bodo.hiframes.pd_series_ext.get_series_name(A)
            out_arr = decode_if_dict_array(arr)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return bodo_decode_if_dict_array_series

    if isinstance(A, bodo.types.DataFrameType):
        if A.is_table_format:
            data_args = "bodo.hiframes.table.decode_if_dict_table(bodo.hiframes.pd_dataframe_ext.get_dataframe_table(A))"
        else:
            # TODO(ehsan): support table format directly to return table format if possible
            data_args = ", ".join(
                f"decode_if_dict_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(A, {i}))"
                for i in range(len(A.columns))
            )
        impl = bodo.hiframes.dataframe_impl._gen_init_df(
            "def impl(A):\n",
            A.columns,
            data_args,
            "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(A)",
            extra_globals={"decode_if_dict_array": decode_if_dict_array, "bodo": bodo},
        )
        return impl

    return lambda A: A  # pragma: no cover


def to_str_arr_if_dict_array(t):
    """convert type 't' to a regular string array if it is a dictionary-encoded array"""
    if t == bodo.types.dict_str_arr_type:
        return bodo.types.string_array_type

    if isinstance(t, types.BaseTuple):
        return types.BaseTuple.from_types(
            [to_str_arr_if_dict_array(a) for a in t.types]
        )

    if isinstance(t, bodo.types.TableType):
        new_arr_types = tuple(to_str_arr_if_dict_array(t) for t in t.arr_types)
        return bodo.types.TableType(new_arr_types, t.has_runtime_cols)

    if isinstance(t, bodo.types.DataFrameType):
        return t.copy(data=tuple(to_str_arr_if_dict_array(t) for t in t.data))

    if isinstance(t, bodo.types.ArrayItemArrayType):
        return bodo.types.ArrayItemArrayType(to_str_arr_if_dict_array(t.dtype))

    if isinstance(t, bodo.types.StructArrayType):
        return bodo.types.StructArrayType(
            tuple(to_str_arr_if_dict_array(a) for a in t.data), t.names
        )

    if isinstance(t, bodo.types.MapArrayType):
        return bodo.types.MapArrayType(
            to_str_arr_if_dict_array(t.key_arr_type),
            to_str_arr_if_dict_array(t.value_arr_type),
        )

    return t


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
        super().__init__(highlight(msg), loc)


class BodoException(numba.core.errors.TypingError):
    """Bodo exception that inherits from numba.core.errors.TypingError
    to allow typing pass to catch it and potentially transform the IR.
    """


class BodoConstUpdatedError(Exception):
    """Indicates that a constant value is expected but the input list/dict/set is
    updated in place. Only used in partial typing pass to enable error checking.
    """


def raise_bodo_error(msg, loc=None) -> typing.NoReturn:
    """Raises BodoException during partial typing in case typing transforms can handle
    the issue. Otherwise, raises BodoError.
    """
    if bodo.transforms.typing_pass.in_partial_typing:
        bodo.transforms.typing_pass.typing_transform_required = True
        raise BodoException(msg)
    else:
        locs = [] if loc is None else [loc]
        raise BodoError(msg, locs_in_msg=locs)


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
    elif hasattr(error, "args") and error.args:
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
        self._concat_str: str | None = None
        # whether _concat_str should be concatenated on the left
        self._concat_left: str | None = None

    def get_schema(self, fname: str):
        """Get dataset schema from file name"""
        full_path = self.get_full_filename(fname)
        return self._get_schema(full_path)

    def set_concat(self, concat_str, is_left):
        """Set input string concatenation parameters"""
        self._concat_str = concat_str
        self._concat_left = is_left

    def _get_schema(self, fname: str) -> FileSchema:
        # should be implemented in subclasses
        raise NotImplementedError

    def get_full_filename(self, fname: str):
        """Get full path with concatenation if necessary"""
        if self._concat_str is None:
            return fname

        if self._concat_left:
            return self._concat_str + fname

        return fname + self._concat_str


class FilenameType(types.Literal):
    """
    Arguments of Bodo functions that are a constant literal are
    converted to this type instead of plain Literal to allow us
    to reuse the cache for differing file names that have the
    same schema. All FilenameType instances have the same hash
    to allow comparison of different instances. Equality is based
    on the schema (not the file name).
    """

    def __init__(self, fname, finfo: FileInfo):
        self.fname = fname
        self._schema = finfo.get_schema(fname)
        super().__init__(self.fname)

    def __hash__(self):
        # fixed number to ensure every FilenameType hashes equally
        return 37

    def __eq__(self, other):
        if isinstance(other, types.FilenameType):
            assert self._schema is not None
            assert other._schema is not None
            # NOTE: check fname type match since the type objects are interned in Numba,
            # and the fact that data model can be either list or string can cause
            # issues if the same type object is reused. See [BE-2050]
            return (bodo.typeof(self.fname) == bodo.typeof(other.fname)) and (
                self._schema == other._schema
            )
        else:
            return False

    @property
    def schema(self):
        # Create a copy in case the contents are mutated.
        return copy.deepcopy(self._schema)


types.FilenameType = FilenameType  # type: ignore

# Data model, unboxing and lower cast are the same as fname (unicode or list) to
# allow passing different file names to compiled code (note that if
# data model is literal the file name would be part of the binary code)
# see test_pq_cache_print


# datamodel
@register_model(types.FilenameType)
class FilenameModel(models.StructModel):
    """FilenameType can hold either a string or a list, so get the fields based on
    value type.
    """

    def __init__(self, dmm, fe_type):
        val_model = dmm.lookup(bodo.typeof(fe_type.fname))
        members = list(zip(val_model._fields, val_model._members))
        super().__init__(dmm, fe_type, members)


@unbox(FilenameType)
def unbox_file_name_type(typ, obj, c):
    return c.unbox(bodo.typeof(typ.fname), obj)


# lower cast
@lower_cast(types.FilenameType, types.unicode_type)
@lower_cast(types.FilenameType, types.List)
def cast_filename_to_unicode(context, builder, fromty, toty, val):
    return val


@box(FilenameType)
def box_filename_type(typ, val, c):
    return c.box(bodo.typeof(typ.fname), val)


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
        or (isinstance(val, types.Omitted) and isinstance(val.value, bool))
    )


def is_overload_bool(val):
    return isinstance(val, types.Boolean) or is_overload_constant_bool(val)


def is_overload_constant_str(val):
    return (
        isinstance(val, str)
        or (isinstance(val, types.StringLiteral) and isinstance(val.literal_value, str))
        or (isinstance(val, types.Omitted) and isinstance(val.value, str))
    )


def is_overload_constant_bytes(val):
    """Checks if the specified value is a binary constant"""
    return (
        isinstance(val, bytes)
        # Numba doesn't have a coresponding literal type for byte literals
        # or (isinstance(val, types.BinaryLiteral) and isinstance(val.literal_value, bytes))
        or (isinstance(val, types.Omitted) and isinstance(val.value, bytes))
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
        isinstance(val, types.Omitted) and isinstance(val.value, float)
    )


def is_overload_int(val):
    return is_overload_constant_int(val) or isinstance(val, types.Integer)


def is_overload_float(val):
    return is_overload_constant_float(val) or isinstance(val, types.Float)


def is_valid_int_arg(arg):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked

    returns: True if the argument is an integer, False otherwise
    """
    return not (
        arg != types.none
        and not isinstance(arg, types.Integer)
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, types.Integer)
        )
        and not is_overload_int(arg)
    )


def is_valid_float_arg(arg):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a float
        (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked

    returns: True if the argument is a float, False otherwise
    """
    return not (
        arg != types.none
        and not isinstance(arg, types.Float)
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, types.Float)
        )
        and not is_overload_float(arg)
    )


def is_overload_numeric_scalar(val):
    return is_overload_bool(val) or is_overload_float(val) or is_overload_int(val)


def is_overload_constant_int(val):
    return (
        isinstance(val, int)
        or (
            isinstance(val, types.IntegerLiteral) and isinstance(val.literal_value, int)
        )
        or (isinstance(val, types.Omitted) and isinstance(val.value, int))
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


def is_overload_const_str_equal(val, const):
    return (
        val == const
        or val == types.StringLiteral(const)
        or getattr(val, "value", -1) == const
    )


def is_overload_str(val):
    return isinstance(val, types.UnicodeType) or is_overload_constant_str(val)


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
    if isinstance(val, bodo.decorators.WrapPythonDispatcherType):
        return val.dispatcher
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


def assert_bodo_error(cond, msg=""):
    """Assertion that raises BodoError instead of regular AssertionError to avoid
    early compiler termination by Numba and allow iterative typing to continue.
    For example, using this for checking for constants allows typing pass to try to
    force the value to be constant.
    """
    if not cond:
        raise BodoError(msg)


def element_type(val):
    """Return the element type of a scalar or array"""
    if isinstance(val, (types.List, types.ArrayCompatible)):
        if isinstance(val.dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
            return val.dtype.elem_type
        # Bytes type is array compatible, but should be treated as scalar
        if val == bodo.types.bytes_type:
            return bodo.types.bytes_type
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


def single_arg_check(v1, v2):
    from bodo.hiframes.datetime_timedelta_ext import _no_input

    return (
        v1 is NOT_CONSTANT
        or (v1 is not None and v2 is None)
        or (v1 is None and v2 is not None)
        or (v1 is not np.nan and v1 != v2)
        or (v1 is np.nan and v2 is not np.nan)
        or (v1 is not np.nan and v2 is np.nan)
        or (v1 is not _no_input and v2 is _no_input)
        or (v1 is _no_input and v2 is not _no_input)
    )


def raise_unsupported_arg(unsupported, package_name, module_name, error_message):
    if unsupported and package_name == "pandas":
        if module_name == "IO":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/io/).\n"
        elif module_name == "General":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/general/).\n"
        elif module_name == "DataFrame":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/dataframe/).\n"
        elif module_name == "Window":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/window/).\n"
        elif module_name == "GroupBy":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/groupby/).\n"
        elif module_name == "Series":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/series/).\n"
        elif module_name == "HeterogeneousSeries":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/series/#heterogeneous_series).\n"
        elif module_name == "Index":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/indexapi/).\n"
        elif module_name == "Timestamp":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/timestamp/).\n"
        elif module_name == "Timedelta":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/timedelta/).\n"
        elif module_name == "DateOffsets":
            error_message += "\nPlease check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/dateoffsets/).\n"

    elif unsupported and package_name == "ml":
        error_message += "\nPlease check supported ML operations here (https://docs.bodo.ai/latest/api_docs/ml/).\n"
    elif unsupported and package_name == "numpy":
        error_message += "\nPlease check supported Numpy operations here (https://docs.bodo.ai/latest/api_docs/numpy/).\n"
    if unsupported:
        raise BodoError(error_message)


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

    assert len(args_dict) == len(arg_defaults_dict)
    if fn_str == None:
        fn_str = f"{fname}()"
    error_message = ""
    unsupported = False
    for a in args_dict:
        v1 = get_overload_const(args_dict[a])
        v2 = arg_defaults_dict[a]
        if single_arg_check(v1, v2):
            error_message = f"{fn_str}: {a} parameter only supports default value {v2}"
            unsupported = True
            break

    raise_unsupported_arg(unsupported, package_name, module_name, error_message)


def get_overload_const_tuple(val) -> tuple | None:
    if isinstance(val, tuple):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, tuple)
        return val.value
    if isinstance(val, types.BaseTuple):
        return tuple(get_overload_const(t) for t in val.types)


def get_overload_constant_dict(val) -> dict:
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


def get_overload_const_list(val) -> list[Any] | tuple[Any, ...] | None:
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


def get_overload_const_str(val) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, str)
        return val.value
    # literal case
    if isinstance(val, types.StringLiteral):
        assert isinstance(val.literal_value, str)
        return val.literal_value
    raise BodoError(f"{val} not constant string")


def get_overload_const_bytes(val) -> bytes:
    """Gets the bytes value from the possibly wraped value.
    Val must actually be a constant byte type, or this fn will throw an error
    """
    if isinstance(val, bytes):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, bytes)
        return val.value
    # Numba has no eqivalent literal type for bytes
    raise BodoError(f"{val} not constant binary")


def get_overload_const_int(val) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, int)
        return val.value
    # literal case
    if isinstance(val, types.IntegerLiteral):
        assert isinstance(val.literal_value, int)
        return val.literal_value
    raise BodoError(f"{val} not constant integer")


def get_overload_const_float(val) -> float:
    if isinstance(val, float):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, float)
        return val.value
    raise BodoError(f"{val} not constant float")


def get_overload_const_bool(val, f_name=None, a_name=None) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, bool)
        return val.value
    # literal case
    if isinstance(val, types.BooleanLiteral):
        assert isinstance(val.literal_value, bool)
        return val.literal_value
    raise BodoError(
        ("" if f_name is None else f"Internal error in {f_name}: ")
        + (
            f"{val} not constant boolean"
            if a_name is None
            else f"{a_name} must be constant boolean, but get {val}"
        )
    )


def is_const_func_type(t) -> bool:
    """check if 't' is a constant function type"""
    return isinstance(
        t,
        (
            types.MakeFunctionLiteral,
            bodo.utils.typing.FunctionLiteral,
            types.Dispatcher,
            bodo.decorators.WrapPythonDispatcherType,
        ),
    )


def get_overload_const_func(val, func_ir):
    """get constant function object or ir.Expr.make_function from function type"""
    from bodo.decorators import WrapPythonDispatcherType

    if isinstance(val, (types.MakeFunctionLiteral, bodo.utils.typing.FunctionLiteral)):
        func = val.literal_value
        # Handle functions that are currently make_function expressions from BodoSQL
        if isinstance(func, ir.Expr) and func.op == "make_function":
            assert_bodo_error(
                func_ir is not None,
                "Function expression is make_function but there is no existing IR",
            )
            func = numba.core.ir_utils.convert_code_obj_to_function(func, func_ir)
        return func
    if isinstance(val, types.Dispatcher):
        return val.dispatcher.py_func
    if isinstance(val, CPUDispatcher):
        return val.py_func

    if isinstance(val, WrapPythonDispatcherType):
        return val.dispatcher

    raise BodoError(f"'{val}' not a constant function type")


def is_heterogeneous_tuple_type(t):
    """check if 't' is a heterogeneous tuple type (or similar, e.g. constant list)"""
    if is_overload_constant_list(t):
        # LiteralList values may be non-constant
        if isinstance(t, types.LiteralList):
            t = types.BaseTuple.from_types(t.types)
        else:
            t = bodo.typeof(tuple(get_overload_const_list(t)))

    if isinstance(t, bodo.types.NullableTupleType):
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
        if dtype.key[0] is float:
            dtype = types.StringLiteral("float")
        elif dtype.key[0] is int:
            dtype = types.StringLiteral("int")
        elif dtype.key[0] is bool:
            dtype = types.StringLiteral("bool")
        elif dtype.key[0] is str:
            dtype = bodo.types.string_type

    # Handle Pandas Int type directly. This can occur when
    # we have a LiteralStrKeyDict so the type is the actual
    # Pandas dtype. See test_table_del_astype.
    if type(dtype) in bodo.libs.int_arr_ext.pd_int_dtype_classes:
        dtype = types.StringLiteral(dtype.name)

    if isinstance(dtype, types.DTypeSpec):
        return dtype.dtype

    # input is array dtype already (see dtype_to_array_type)
    if isinstance(
        dtype,
        (
            types.Number,
            types.NPDatetime,
            bodo.types.TimestampTZType,
            bodo.types.Decimal128Type,
            bodo.types.StructType,
            bodo.types.MapScalarType,
            bodo.types.TimeType,
            bodo.hiframes.pd_categorical_ext.PDCategoricalDtype,
            bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype,
        ),
    ) or dtype in (
        bodo.types.string_type,
        bodo.types.bytes_type,
        bodo.types.datetime_date_type,
        bodo.types.datetime_timedelta_type,
        bodo.types.null_dtype,
        bodo.types.pd_timestamp_tz_naive_type,
        bodo.types.pd_timedelta_type,
    ):
        return dtype

    try:
        d_str = get_overload_const_str(dtype)
        if d_str.startswith("Int") or d_str.startswith("UInt"):
            return bodo.libs.int_arr_ext.typeof_pd_int_dtype(
                pd.api.types.pandas_dtype(d_str), None
            )
        if d_str.startswith("Float"):
            return bodo.libs.float_arr_ext.typeof_pd_float_dtype(
                pd.api.types.pandas_dtype(d_str), None
            )
        if d_str == "boolean":
            return bodo.libs.bool_arr_ext.boolean_dtype
        if d_str == "str":
            return bodo.types.string_type

        # Handle separately since Numpy < 2 on Windows returns int32 in np.dtype
        if d_str == "int":
            return types.int64

        return numba.np.numpy_support.from_dtype(np.dtype(d_str))
    except Exception:
        pass
    if func_name is not None:
        raise BodoError(f"{func_name}(): invalid dtype {dtype}")
    else:
        raise BodoError(f"invalid dtype {dtype}")


def is_list_like_index_type(
    t,
) -> bool:
    """Types that can be similar to list for indexing Arrays, Series, etc.
    Tuples are excluded due to indexing semantics.
    """
    from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.libs.bool_arr_ext import boolean_array_type

    # TODO: include datetimeindex/timedeltaindex?

    return (
        isinstance(t, types.List)
        or (isinstance(t, types.Array) and t.ndim == 1)
        or isinstance(t, (NumericIndexType, RangeIndexType))
        or isinstance(t, SeriesType)
        or isinstance(t, bodo.types.IntegerArrayType)
        or t == boolean_array_type
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

    err_msg = f"{func_name}: index name should be a constant string"

    # MultiIndex has multiple names
    if isinstance(t, MultiIndexType):
        names = []
        for i, n_typ in enumerate(t.names_typ):
            if n_typ == types.none:
                names.append(f"level_{i}")
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


def to_numeric_index_if_range_index(t):
    """Convert RangeIndexType to NumericIndexType (if input is RangeIndexType)

    Args:
        t (types.Type): input Index type

    Returns:
        types.Type: same input if not RangeIndexType, otherwise NumericIndexType
    """
    from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType

    return (
        NumericIndexType(types.int64, t.name_typ)
        if isinstance(t, RangeIndexType)
        else t
    )


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

    if t in [
        bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type,
        bodo.types.datetime64ns,
    ]:
        return DatetimeIndexType(types.none)

    # Timezone-aware timestamp
    if (
        isinstance(t, bodo.hiframes.pd_timestamp_ext.PandasTimestampType)
        and t.tz is not None
    ):
        return DatetimeIndexType(
            types.none, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType(t.tz)
        )

    if t in [
        bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type,
        bodo.types.timedelta64ns,
    ]:
        return TimedeltaIndexType(types.none)

    if t == bodo.types.string_type:
        return StringIndexType(types.none)

    if t == bodo.types.bytes_type:
        return BinaryIndexType(types.none)

    if (
        isinstance(t, (types.Integer, types.Float, types.Boolean))
        or t == bodo.types.datetime_date_type
    ):
        return NumericIndexType(t, types.none)

    if isinstance(t, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        return CategoricalIndexType(bodo.types.CategoricalArrayType(t))

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


def is_bodosql_context_type(t):
    """Check for BodoSQLContextType without importing BodoSQL unnecessarily"""
    if type(t).__name__ == "BodoSQLContextType":
        try:
            import bodosql.compiler  # isort:skip # noqa
            from bodosql.context_ext import BodoSQLContextType
        except ImportError:  # pragma: no cover
            raise ImportError("BodoSQL not installed properly")
        assert isinstance(t, BodoSQLContextType), (
            "is_bodosql_context_type: expected BodoSQLContextType"
        )
        return True

    return False


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
# typed.Dict tries to get the type of the UDF value, which is not possible. This hack
# makes a dummy type available to Numba so that type inference works.
types.MakeFunctionLiteral._literal_type_cache = types.MakeFunctionLiteral(lambda: 0)


def _get_key_bool_safe(meta):
    """Convert bool values to string to use as key since values like (True, False) and
    (1, 0) are equal in Python, but shouldn't be equal in this context.
    This causes Numba's type instance interning to assume instances are the same
    which is wrong.
    See https://bodo.atlassian.net/browse/BSE-2809

    Args:
        meta (any): input meta value to convert to key (typically tuple of string or int or bool)

    Returns:
        any: meta value with bools replaced with string key
    """
    if isinstance(meta, bool):
        return f"_$BODO_BOOL_{meta}"
    if isinstance(meta, tuple):
        return tuple(_get_key_bool_safe(a) for a in meta)

    return meta


# type used to pass metadata to type inference functions
# see untyped_pass.py and df.pivot_table()
class MetaType(types.Type):
    def __init__(self, meta):
        # TODO: this may not work for custom types, checking __hash__ attribute exists
        # may be better, but I'm uncertain if that's correct either
        if not isinstance(meta, typing.Hashable):  # pragma: no cover
            raise RuntimeError("Internal error: MetaType should be hashable")
        self.meta = meta
        super().__init__(f"MetaType({meta})")

    def can_convert_from(self, typingctx, other):
        return True

    @property
    def key(self):
        return _get_key_bool_safe(self.meta)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    def __len__(self):
        # determine len based on the meta values.
        return len(self.meta)


register_model(MetaType)(models.OpaqueModel)


# A subclass of MetaType that is used to pass around column names
# This has no differences with MetaType, it exists purely to make the code more readable
class ColNamesMetaType(MetaType):
    def __init__(self, meta):
        if not isinstance(meta, typing.Hashable):  # pragma: no cover
            raise RuntimeError("Internal error: ColNamesMetaType should be hashable")
        self.meta = meta
        types.Type.__init__(self, f"ColNamesMetaType({meta})")


register_model(ColNamesMetaType)(models.OpaqueModel)


# A subclass of MetaType that is used to pass around information
# when creating a table in Snowflake
class CreateTableMetaType(MetaType):
    def __init__(self, table_comment=None, column_comments=None, table_properties=None):
        self.table_comment = table_comment
        self.column_comments = column_comments
        self.table_properties = table_properties
        meta = (self.table_comment, self.column_comments, self.table_properties)
        if not isinstance(meta, typing.Hashable):  # pragma: no cover
            raise RuntimeError("Internal error: ColNamesMetaType should be hashable")
        self.meta = meta
        types.Type.__init__(self, f"CreateTableMetaType({meta})")


register_model(CreateTableMetaType)(models.OpaqueModel)


def is_literal_type(t):
    """return True if 't' represents a data type with known compile-time constant value"""
    return (
        isinstance(t, types.TypeRef)
        # LiteralStrKeyDict is not always a literal since its values are not necessarily
        # constant
        or (
            isinstance(t, (types.Literal, types.Omitted))
            and not isinstance(t, types.LiteralStrKeyDict)
        )
        or t == types.none  # None type is always literal since single value
        or isinstance(t, types.Dispatcher)
        or isinstance(t, bodo.decorators.WrapPythonDispatcherType)
        # LiteralStrKeyDict is a BaseTuple in Numba 0.51 also
        or (isinstance(t, types.BaseTuple) and all(is_literal_type(v) for v in t.types))
        # List/Dict types preserve const initial values in Numba 0.51
        or is_initial_value_type(t)
        # dtype literals should be treated as literals
        or isinstance(t, (types.DTypeSpec, types.Function))
        or isinstance(t, bodo.libs.int_arr_ext.IntDtype)
        or isinstance(t, bodo.libs.float_arr_ext.FloatDtype)
        or t
        in (
            bodo.libs.bool_arr_ext.boolean_dtype,
            bodo.libs.str_arr_ext.string_dtype,
            bodo.types.datetime64ns,
            bodo.types.timedelta64ns,
        )
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
        and is_literal_type(t.name_typ)
    )


def get_overload_constant_index(t):
    """return compile time constant value for Index type 't' (assuming it is a literal)"""
    assert is_overload_constant_index(t)
    return pd.Index(get_literal_value(t.data), name=get_literal_value(t.name_typ))


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
    assert_bodo_error(is_literal_type(t))
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
    if isinstance(t, bodo.decorators.WrapPythonDispatcherType):
        return t.dispatcher
    if is_initial_value_type(t):
        return t.initial_value
    if isinstance(t, (types.DTypeSpec, types.Function)):
        return t
    if isinstance(t, bodo.libs.int_arr_ext.IntDtype):
        return getattr(pd, str(t)[:-2])()
    if isinstance(t, bodo.libs.float_arr_ext.FloatDtype):  # pragma: no cover
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
        t in (bodo.types.string_type, types.bool_)
        or isinstance(t, (types.Integer, types.List, types.SliceType, types.DictType))
        or (pyobject_to_literal and t == types.pyobject)
    )


def dtype_to_array_type(dtype, convert_nullable=False):
    """get default array type for scalar dtype

    Args:
        dtype (types.Type): scalar data type
    """
    dtype = types.unliteral(dtype)

    # UDFs may use Optional types for setting array values.
    # These should use the nullable type of the non-null case
    if isinstance(dtype, types.Optional):
        dtype = dtype.type
        convert_nullable = True

    # UDFs may return lists, but we store array of array for output
    if isinstance(dtype, types.List):
        dtype = dtype_to_array_type(dtype.dtype, convert_nullable)

    # null array
    if dtype == bodo.types.null_dtype or dtype == bodo.types.none:
        return bodo.types.null_array_type

    # string array
    if dtype == bodo.types.string_type:
        return bodo.types.string_array_type

    # binary array
    if dtype == bodo.types.bytes_type:
        return bodo.types.binary_array_type

    if bodo.utils.utils.is_array_typ(dtype, False):
        return bodo.types.ArrayItemArrayType(dtype)

    # categorical
    if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        return bodo.types.CategoricalArrayType(dtype)

    if isinstance(dtype, bodo.libs.int_arr_ext.IntDtype):
        return bodo.types.IntegerArrayType(dtype.dtype)

    if isinstance(dtype, bodo.libs.float_arr_ext.FloatDtype):  # pragma: no cover
        return bodo.types.FloatingArrayType(dtype.dtype)

    if dtype == types.boolean:
        return bodo.types.boolean_array_type

    if dtype == bodo.types.datetime_date_type:
        return bodo.hiframes.datetime_date_ext.datetime_date_array_type

    if isinstance(dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype):
        return bodo.libs.pd_datetime_arr_ext.DatetimeArrayType(dtype.tz)

    if isinstance(dtype, bodo.types.TimeType):
        return bodo.hiframes.time_ext.TimeArrayType(dtype.precision)

    if dtype == bodo.types.timestamptz_type:
        return bodo.hiframes.timestamptz_ext.timestamptz_array_type

    if isinstance(dtype, bodo.types.Decimal128Type):
        return bodo.types.DecimalArrayType(dtype.precision, dtype.scale)

    # struct array
    if isinstance(dtype, bodo.libs.struct_arr_ext.StructType):
        return bodo.types.StructArrayType(
            tuple(dtype_to_array_type(t, True) for t in dtype.data), dtype.names
        )

    # tuple array
    if isinstance(dtype, types.BaseTuple):
        return bodo.types.TupleArrayType(
            tuple(dtype_to_array_type(t, convert_nullable) for t in dtype.types)
        )

    # map array
    if isinstance(dtype, bodo.libs.map_arr_ext.MapScalarType):
        return bodo.types.MapArrayType(
            dtype.key_arr_type,
            dtype.value_arr_type,
        )

    if isinstance(dtype, types.DictType):
        return bodo.types.MapArrayType(
            dtype_to_array_type(dtype.key_type, convert_nullable),
            dtype_to_array_type(dtype.value_type, convert_nullable),
        )

    # DatetimeTZDtype are stored as pandas Datetime array
    if isinstance(dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype):
        return bodo.types.DatetimeArrayType(dtype.tz)

    if isinstance(dtype, bodo.types.PandasTimestampType) and dtype.tz is not None:
        return bodo.types.DatetimeArrayType(dtype.tz)

    # Timestamp/datetime are stored as dt64 array
    if dtype in (
        bodo.types.pd_timestamp_tz_naive_type,
        bodo.hiframes.datetime_datetime_ext.datetime_datetime_type,
    ):
        return types.Array(bodo.types.datetime64ns, 1, "C")

    # pd.Timedelta/datetime.timedelta values are stored as td64 arrays
    if dtype in (
        bodo.types.pd_timedelta_type,
        bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type,
    ):
        return types.Array(bodo.types.timedelta64ns, 1, "C")

    # regular numpy array
    if isinstance(dtype, (types.Number, types.NPDatetime, types.NPTimedelta)):
        arr = types.Array(dtype, 1, "C")
        # If this comes from an optional type try converting to
        # nullable.
        if convert_nullable:
            return to_nullable_type(arr)
        return arr
    if isinstance(dtype, bodo.types.MapScalarType):
        return bodo.types.MapArrayType(dtype.key_arr_type, dtype.value_arr_type)
    raise BodoError(f"dtype {dtype} cannot be stored in arrays")  # pragma: no cover


def get_udf_out_arr_type(f_return_type, return_nullable=False):
    """get output array type of a UDF call, give UDF's scalar output type.
    E.g. S.map(lambda a: 2) -> array(int64)
    """

    # UDF output can be Optional if None is returned in a code path
    if isinstance(f_return_type, types.Optional):
        f_return_type = f_return_type.type
        return_nullable = True

    # Needed for MapArrayType since we use MapScalarType in getitem, not Dict
    # See test_map_array.py::test_map_apply_simple
    if isinstance(f_return_type, types.DictType):
        return_nullable = True

    # unbox Timestamp to dt64 in Series
    if f_return_type == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type:
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
    except Exception:
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
        bodo.hiframes.pd_timestamp_ext.PandasTimestampType,
    )
    whitelist_instances = (
        types.bool_,
        bodo.types.datetime64ns,
        bodo.types.timedelta64ns,
        bodo.types.pd_timedelta_type,
    )

    if isinstance(t, whitelist_types) or (t in whitelist_instances):
        return True

    typing_context = numba.core.registry.cpu_target.typing_context
    try:
        typing_context.resolve_function_type(hash, (t,), {})
        return True
    except Exception:  # pragma: no cover
        return False


def to_nullable_type(t):
    """Converts types that cannot hold NAs to corresponding nullable types.
    For example, boolean_array_type is returned for Numpy array(bool_) and IntegerArray is
    returned for Numpy array(int).
    Converts data in DataFrame and Series types as well.
    """
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.hiframes.pd_index_ext import NumericIndexType
    from bodo.hiframes.pd_series_ext import SeriesType

    if isinstance(t, DataFrameType):
        new_data = tuple(to_nullable_type(t) for t in t.data)
        return DataFrameType(new_data, t.index, t.columns, t.dist, t.is_table_format)

    if isinstance(t, SeriesType):
        return SeriesType(
            t.dtype, to_nullable_type(t.data), t.index, t.name_typ, t.dist
        )

    if isinstance(t, NumericIndexType):
        return NumericIndexType(t.dtype, t.name_typ, to_nullable_type(t.data))

    if isinstance(t, types.Array):
        if t.dtype == types.bool_:
            return bodo.libs.bool_arr_ext.boolean_array_type

        if isinstance(t.dtype, types.Integer):
            return bodo.libs.int_arr_ext.IntegerArrayType(t.dtype)

        if isinstance(t.dtype, types.Float):
            return bodo.libs.float_arr_ext.FloatingArrayType(t.dtype)

    if isinstance(t, bodo.types.ArrayItemArrayType):
        return bodo.types.ArrayItemArrayType(to_nullable_type(t.dtype))

    if isinstance(t, bodo.types.StructArrayType):
        return bodo.types.StructArrayType(
            tuple(to_nullable_type(a) for a in t.data), t.names
        )

    if isinstance(t, bodo.types.MapArrayType):
        return bodo.types.MapArrayType(
            to_nullable_type(t.key_arr_type), to_nullable_type(t.value_arr_type)
        )

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
            (
                SeriesType,
                DataFrameType,
                types.List,
                types.BaseTuple,
                types.LiteralList,
                types.RangeType,
            ),
        )
        or bodo.hiframes.pd_index_ext.is_pd_index_type(t)
    )


def is_scalar_type(t: types.Type) -> bool:
    """
    Returns True if 't' is a scalar type like integer, boolean, string, ...
    """
    return isinstance(
        t,
        (
            types.Boolean,
            types.Number,
            types.IntegerLiteral,
            types.BooleanLiteral,
            types.StringLiteral,
            bodo.hiframes.pd_timestamp_ext.PandasTimestampType,
            bodo.types.TimeType,
            bodo.types.Decimal128Type,
        ),
    ) or t in (
        bodo.types.datetime64ns,
        bodo.types.timedelta64ns,
        bodo.types.string_type,
        bodo.types.bytes_type,
        bodo.types.datetime_date_type,
        bodo.types.datetime_datetime_type,
        bodo.types.datetime_timedelta_type,
        bodo.types.pd_timedelta_type,
        bodo.types.month_end_type,
        bodo.types.week_type,
        bodo.types.date_offset_type,
        types.none,
        bodo.types.null_dtype,
        bodo.types.timestamptz_type,
    )


def is_common_scalar_dtype(scalar_types):
    """Returns True if a list of scalar types share a common
    Numpy type or are equal.
    """
    common_type, _ = get_common_scalar_dtype(scalar_types)
    return common_type is not None


# Number of significant digits in every major integer size
# Keys are Numba integer scalar types
# Values are the number of significant digits (in base 10) that always fit
# in the specified integer type
# Always assuming signed integers
SIGS_IN_INT = {
    types.int64: 18,
    types.int32: 9,
    types.int16: 4,
    types.int8: 2,
}


def get_common_scalar_dtype(
    scalar_types: list[types.Type],
    allow_downcast: bool = False,
) -> tuple[types.Type | None, bool]:
    """
    Attempts to unify the list of passed in dtypes, notifying if a downcast
    has occurred.

    Args:
        scalar_types: All dtypes to unify
        allow_downcast: Whether to allow for downcasts, notifying if it occurs.
            If false, will not allow downcasts and just return None, False

    Returns:
        types.Type | None: Unified Dtype or None if not possible
        bool: Whether a downcast has occurred or not (always False when allow_downcast=False)
    """
    scalar_types = [types.unliteral(a) for a in scalar_types]

    if len(scalar_types) == 0:
        raise_bodo_error(
            "Internal error, length of argument passed to get_common_scalar_dtype scalar_types is 0"
        )
    if all(t == bodo.types.null_dtype for t in scalar_types):
        return (bodo.types.null_dtype, False)
    # bodo.types.null_dtype can be cast to any type so remove it from the list.
    scalar_types = [t for t in scalar_types if t != bodo.types.null_dtype]
    try:
        common_dtype = np.result_type(
            *[numba.np.numpy_support.as_dtype(t) for t in scalar_types]
        )
        # If we get an object dtype we do not have a common type.
        # Otherwise, the types can be used together
        if common_dtype is not object:
            return (numba.np.numpy_support.from_dtype(common_dtype), False)

    # If we have a Bodo or Numba type that isn't implemented in
    # Numpy, we will get a NumbaNotImplementedError
    except numba.core.errors.NumbaNotImplementedError:
        pass
    # If we get types that aren't compatible in Numpy, we will get a
    # DTypePromotionError
    except np.exceptions.DTypePromotionError:
        pass

    # Timestamp/dt64 can be used interchangeably
    # TODO: Should datetime.datetime also be included?
    if all(
        t
        in (
            bodo.types.datetime64ns,
            bodo.types.pd_timestamp_tz_naive_type,
            bodo.types.pd_datetime_tz_naive_type,
        )
        for t in scalar_types
    ):
        return (bodo.types.pd_datetime_tz_naive_type, False)

    if all(
        t
        in (
            bodo.types.timedelta64ns,
            bodo.types.pd_timedelta_type,
        )
        for t in scalar_types
    ):
        return (bodo.types.timedelta64ns, False)

    # Datetime+timezone-aware and timestamp+timezone-aware can be converted to be the same
    # if they both have the same timezone value.
    if all(
        isinstance(t, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype)
        or (isinstance(t, bodo.types.PandasTimestampType) and t.tz is not None)
        for t in scalar_types
    ):
        timezones = [t.tz for t in scalar_types]
        for tz in timezones:
            if tz != timezones[0]:
                return (None, False)
        return (bodo.types.PandasTimestampType(timezones[0]), False)
    # If all are Numeric types and one is Decimal128Type, then:
    # - We attempt to combine lossless-ly and reduce to closest non-Decimal type
    # - If too large, we default to closes Decimal128 type expecting lossy conversion
    if any(isinstance(t, bodo.types.Decimal128Type) for t in scalar_types):
        if any(
            not isinstance(t, (types.Number, bodo.types.Decimal128Type))
            for t in scalar_types
        ):
            return None, False

        # First, determine the max # of digits needed to store the
        # digits before and after the decimal place
        # Only for Decimal and Integers, Floats are handled after
        max_float = None
        num_before_digits = 0
        scale = 0

        for t in scalar_types:
            if isinstance(t, types.Float):
                max_float = t if max_float is None else max(max_float, t)
            elif isinstance(t, types.Integer):
                num_before_digits = max(num_before_digits, SIGS_IN_INT[t])
            else:
                assert isinstance(t, bodo.types.Decimal128Type)
                num_before_digits = max(num_before_digits, t.precision - t.scale)
                scale = max(scale, t.scale)

        precision = num_before_digits + scale
        # Precision can be at most 38
        # TODO: What to do if precision > 38. For example, with input Decimal128(38, 0) and Decimal128(38, 18)
        if precision > 38:
            out = (
                types.float64
                if max_float is not None
                else bodo.types.Decimal128Type(38, scale)
            )
            return (out, True) if allow_downcast else (None, False)
        elif precision <= 18 and scale == 0:
            if precision <= 2:
                base_out = types.int8
            elif precision <= 4:
                base_out = types.int16
            elif precision <= 9:
                base_out = types.int32
            else:
                base_out = types.int64
        # 23 bits for float32 mantissa -> 6 sig figs
        elif precision <= 6:
            base_out = types.float32
        # 52 bits for float64 mantissa -> 15 sig figs
        elif precision <= 15:
            base_out = types.float64
        else:
            base_out = bodo.types.Decimal128Type(precision, scale)

        if max_float is None:
            return (base_out, False)

        if allow_downcast and isinstance(base_out, bodo.types.Decimal128Type):
            return (types.float64, True)

        # Combine max_float (float) and base_out (float or int) types
        return get_common_scalar_dtype([max_float, base_out])

    # If we have a mix of MapScalarTypes and DictTypes, then we first convert
    # all DictTypes to MapScalarType.
    if any(isinstance(t, types.DictType) for t in scalar_types) and any(
        isinstance(t, bodo.types.MapScalarType) for t in scalar_types
    ):
        new_types = []
        for t in scalar_types:
            if isinstance(t, types.DictType):
                equivalent_map_type = bodo.types.MapScalarType(
                    dtype_to_array_type(t.key_type), dtype_to_array_type(t.value_type)
                )
                new_types.append(equivalent_map_type)
            else:
                new_types.append(t)
        scalar_types = new_types

    # MapScalarType types are combinable if their key types and value types are
    # also combinable.
    if all(isinstance(t, bodo.types.MapScalarType) for t in scalar_types):
        key_type, key_downcast = get_common_scalar_dtype(
            [t.key_arr_type.dtype for t in scalar_types]
        )
        val_type, val_downcast = get_common_scalar_dtype(
            [t.value_arr_type.dtype for t in scalar_types]
        )
        if key_type is None or val_type is None:
            return (None, False)
        key_arr_type = dtype_to_array_type(key_type)
        val_arr_type = dtype_to_array_type(val_type)
        return (
            bodo.types.MapScalarType(key_arr_type, val_arr_type),
            key_downcast or val_downcast,
        )

    # Dict types are combinable if their key types and value types
    # are also combinable.
    if all(isinstance(t, types.DictType) for t in scalar_types):
        key_type, key_downcast = get_common_scalar_dtype(
            [t.key_type for t in scalar_types]
        )
        val_type, val_downcast = get_common_scalar_dtype(
            [t.value_type for t in scalar_types]
        )
        if key_type is None or val_type is None:
            return (None, False)
        return (types.DictType(key_type, val_type), key_downcast or val_downcast)

    # Struct types are combinable if they have the same field names, and each field type
    # is combinable with the same filed in all the other types.
    if all(isinstance(t, bodo.libs.struct_arr_ext.StructType) for t in scalar_types):
        names = scalar_types[0].names
        if not all(t.names == names for t in scalar_types):
            return None, False
        new_field_types = []
        downcasted = False
        for i in range(len(names)):
            inner_types = [t.data[i] for t in scalar_types]
            common_dtype, downcast = get_common_scalar_dtype(inner_types)
            if common_dtype is None:
                return (None, False)
            new_field_types.append(common_dtype)
            downcasted = downcasted or downcast
        return (
            bodo.libs.struct_arr_ext.StructType(tuple(new_field_types), names),
            downcasted,
        )

    # If we don't have a common type, then all types need to be equal.
    # See: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    grouped_types = itertools.groupby(scalar_types)
    if next(grouped_types, True) and not next(grouped_types, False):
        return (scalar_types[0], False)

    return (None, False)


def find_common_np_dtype(arr_types):
    """finds common numpy dtype of array types using np.result_type"""
    try:
        return numba.np.numpy_support.from_dtype(
            np.result_type(
                *[numba.np.numpy_support.as_dtype(t.dtype) for t in arr_types]
            )
        )
    # If we have a Bodo or Numba type that isn't implemented in
    # Numpy, we will get a NumbaNotImplementedError
    except numba.core.errors.NumbaNotImplementedError:
        raise_bodo_error(f"Unable to find a common dtype for types: {arr_types}")
    # If we get types that aren't compatible in Numpy, we will get a
    # DTypePromotionError
    except np.exceptions.DTypePromotionError:
        raise_bodo_error(f"Unable to find a common dtype for types: {arr_types}")


def is_immutable(typ: types.Type) -> bool:
    """
    Returns True if typ is an immutable type, like a scalar or
    tuple of immutable types
    """
    if is_tuple_like_type(typ):
        return all(is_immutable(t) for t in typ.types)
    return is_scalar_type(typ)


def is_immutable_array(typ):
    """
    Returns if typ is an immutable array types. This is used for setitem
    error checking.
    """
    return isinstance(
        typ,
        (
            bodo.types.ArrayItemArrayType,
            bodo.types.MapArrayType,
        ),
    )


def get_nullable_and_non_nullable_types(array_of_types):
    """For each (non-)nullable type in the input list, add the corresponding (non)-nullable
    types to the list and return it. This makes checks for types more robust,
    specifically in pd.DataFrame.select_dtypes func."""

    all_types = []
    for typ in array_of_types:
        if typ == bodo.libs.bool_arr_ext.boolean_array_type:
            all_types.append(types.Array(types.bool_, 1, "C"))

        elif isinstance(
            typ,
            (
                bodo.libs.int_arr_ext.IntegerArrayType,
                bodo.libs.float_arr_ext.FloatingArrayType,
            ),
        ):
            all_types.append(types.Array(typ.dtype, 1, "C"))

        elif isinstance(typ, types.Array):
            if typ.dtype == types.bool_:
                all_types.append(bodo.libs.bool_arr_ext.boolean_array_type)

            if isinstance(typ.dtype, types.Integer):
                all_types.append(bodo.libs.int_arr_ext.IntegerArrayType(typ.dtype))

            if isinstance(typ.dtype, types.Float):
                all_types.append(bodo.libs.float_arr_ext.FloatingArrayType(typ.dtype))

        all_types.append(typ)

    return all_types


def is_np_arr_typ(t, dtype, ndim=1):
    """return True if t is a Numpy array type with the given dtype and ndim. Ignores
    other types.Array flags like 'mutable'
    """
    return isinstance(t, types.Array) and t.dtype == dtype and t.ndim == ndim


def _gen_objmode_overload(
    func, output_type, attr_name=None, is_function=True, single_rank=False
):
    """code gen for gen_objmode_func_overload and gen_objmode_method_overload"""
    if is_function:
        func_spec = getfullargspec(func)

        assert func_spec.varargs is None, "varargs not supported"
        assert func_spec.varkw is None, "varkw not supported"

        defaults = [] if func_spec.defaults is None else func_spec.defaults
        n_pos_args = len(func_spec.args) - len(defaults)

        # Matplotlib specifies some arguments as `<deprecated parameter>`.
        # We can't support them, and it breaks our infrastructure, so omit them.
        #
        def get_default(default_val):
            if isinstance(default_val, str):
                return "'" + default_val + "'"
            else:
                return str(default_val)

        args = func_spec.args[1:] if attr_name else func_spec.args[:]
        arg_strs = []
        for i, arg in enumerate(func_spec.args):
            if i < n_pos_args:
                arg_strs.append(arg)
            elif str(defaults[i - n_pos_args]) != "<deprecated parameter>":
                arg_strs.append(arg + "=" + get_default(defaults[i - n_pos_args]))
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
    else:
        sig = "self"

    # workaround objmode string type name requirement by adding the type to types module
    # TODO: fix Numba's object mode to take type refs
    type_name = str(output_type)
    if not hasattr(types, type_name):
        type_name = f"objmode_type{ir_utils.next_label()}"
        setattr(types, type_name, output_type)

    if not attr_name:
        # This Python function is going to be set at the global scope of this
        # module (bodo.utils.typing) so we need a name that won't clash
        func_name = func.__module__.replace(".", "_") + "_" + func.__name__ + "_func"

    call_str = f"self.{attr_name}" if attr_name else f"{func_name}"
    func_text = f"def overload_impl({sig}):\n"
    func_text += f"    def impl({sig}):\n"
    if single_rank:
        func_text += "        if bodo.get_rank() == 0:\n"
        extra_indent = "    "
    else:
        extra_indent = ""
    # TODO: Should we add a parameter to avoid the objmode warning?
    func_text += f"        {extra_indent}with numba.objmode(res='{type_name}'):\n"
    if is_function:
        func_text += f"            {extra_indent}res = {call_str}({args})\n"
    else:
        func_text += f"            {extra_indent}res = {call_str}\n"
    func_text += "        return res\n"
    func_text += "    return impl\n"

    loc_vars = {}
    # XXX For some reason numba needs a reference to the module or caching
    # won't work (and seems related to objmode).
    glbls = globals()
    if not attr_name:
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
            func, output_type, is_function=True, single_rank=single_rank
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
            method, output_type, method_name, True, single_rank
        )
        overload_method(obj_type, method_name, no_unliteral=True)(overload_impl)
    except Exception:
        # If the module has changed in a way we can't support (i.e. varargs in matplotlib),
        # then don't do the overload
        pass


def gen_objmode_attr_overload(
    obj_type, attr_name, attr, output_type=None, single_rank=False
):
    try:
        overload_impl = _gen_objmode_overload(
            attr, output_type, attr_name, False, single_rank
        )
        overload_attribute(obj_type, attr_name, no_unliteral=True)(overload_impl)
    except Exception:  # pragma: no cover
        # This is preserved from the func/method objmode overloads to handle unsupported
        # changes (e.g. varargs), this is likely not stricly necessary here.
        pass


@infer
class NumTypeStaticGetItem(AbstractTemplate):
    """typer for getitem on number types in JIT code
    e.g. bodo.types.int64[::1] -> array(int64, 1, "C")
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
def unliteral_val(typingctx, val):
    """converts the type of value 'val' to nonliteral"""

    def codegen(context, builder, signature, args):
        return args[0]

    return types.unliteral(val)(val), codegen


def create_unsupported_overload(fname):
    """Create an overload for unsupported function 'fname' that raises BodoError"""

    def overload_f(*a, **kws):
        raise BodoError(f"{fname} not supported yet")

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


def is_numpy_function(func):
    """
    Determine if func is a builtin function from NumPy.
    """
    return isinstance(func, types.Function) and func.typing_key.__module__ == "numpy"


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
    func_text = "def stub("
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
                assert arg_name in defaults, (
                    f"{func_name}(): '{arg_name}' is unsupported but no default is provided"
                )
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
        return bodo.types.StringIndexType(None)
    elif all(isinstance(a, bytes) for a in col_names):
        return bodo.types.BinaryIndexType(None)
    elif all(isinstance(a, (int, float)) for a in col_names):  # pragma: no cover
        # TODO(ehsan): test
        if any(isinstance(a, (float)) for a in col_names):
            return bodo.types.NumericIndexType(types.float64)
        else:
            return bodo.types.NumericIndexType(types.int64)
    else:
        return bodo.hiframes.pd_index_ext.HeterogeneousIndexType(
            bodo.typeof(tuple(types.literal(c) for c in col_names))
        )


class BodoArrayIterator(types.SimpleIteratorType):
    """
    Type class for iterators of bodo arrays.
    TODO(ehsan): add iterator support using this for all bodo array types.
    """

    def __init__(self, arr_type, yield_type=None):
        self.arr_type = arr_type
        name = f"iter({arr_type})"
        if yield_type == None:
            yield_type = arr_type.dtype
        super().__init__(name, yield_type)


@register_model(BodoArrayIterator)
class BodoArrayIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [
            ("index", types.EphemeralPointer(types.uintp)),
            ("array", fe_type.arr_type),
        ]
        super().__init__(dmm, fe_type, members)


@lower_builtin("iternext", BodoArrayIterator)
@iternext_impl(RefType.NEW)
def iternext_bodo_array(context, builder, sig, args, result):
    [iterty] = sig.args
    [iter_arg] = args

    iterobj = context.make_helper(builder, iterty, value=iter_arg)
    len_sig = signature(types.intp, iterty.arr_type)
    nitems = context.compile_internal(
        builder, lambda a: len(a), len_sig, [iterobj.array]
    )

    index = builder.load(iterobj.index)
    is_valid = builder.icmp_signed("<", index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        getitem_sig = signature(iterty.yield_type, iterty.arr_type, types.intp)
        value = context.compile_internal(
            builder, lambda a, i: a[i], getitem_sig, [iterobj.array, index]
        )
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


def index_typ_from_dtype_name_arr(elem_dtype, name, arr_typ):
    """
    Given a dtype, name (which is either None or a string),
    and possibly an array type, returns a matching index type.
    """
    index_class = type(get_index_type_from_dtype(elem_dtype))
    if name is None:
        name_typ = None
    elif name == types.none or isinstance(name, types.StringLiteral):
        name_typ = name
    else:
        name_typ = types.StringLiteral(name)
    if index_class == bodo.hiframes.pd_index_ext.NumericIndexType:
        # Numeric requires the size
        index_typ = index_class(elem_dtype, name_typ, arr_typ)
    elif index_class == bodo.hiframes.pd_index_ext.CategoricalIndexType:
        # Categorical requires the categorical array
        index_typ = index_class(
            bodo.types.CategoricalArrayType(elem_dtype), name_typ, arr_typ
        )
    else:
        index_typ = index_class(name_typ, arr_typ)
    return index_typ


def is_safe_arrow_cast(lhs_scalar_typ, rhs_scalar_typ):
    """
    Determine if two scalar types which return False from
    'is_common_scalar_dtype' can be safely cast in an arrow
    filter expression. This is a white list of casts that
    are manually supported.
    """
    # TODO: Support more types
    # All tests except lhs: date are currently marked as slow
    if lhs_scalar_typ == types.unicode_type:  # pragma: no cover
        # Cast is supported between string and timestamp
        return rhs_scalar_typ in (
            bodo.types.datetime64ns,
            bodo.types.pd_timestamp_tz_naive_type,
        )
    elif rhs_scalar_typ == types.unicode_type:  # pragma: no cover
        # Cast is supported between timestamp and string
        return lhs_scalar_typ in (
            bodo.types.datetime64ns,
            bodo.types.pd_timestamp_tz_naive_type,
        )
    elif lhs_scalar_typ == bodo.types.datetime_date_type:
        # Cast is supported between date and timestamp
        return rhs_scalar_typ in (
            bodo.types.datetime64ns,
            bodo.types.pd_timestamp_tz_naive_type,
        )
    elif rhs_scalar_typ == bodo.types.datetime_date_type:  # pragma: no cover
        # Cast is supported between date and timestamp
        return lhs_scalar_typ in (
            bodo.types.datetime64ns,
            bodo.types.pd_timestamp_tz_naive_type,
        )
    return False  # pragma: no cover


def register_type(type_name, type_value):
    """register a data type to be used in objmode blocks"""
    import bodo.spawn.spawner

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

    # TODO[BSE-4170]: simplify test flags
    if bodo.spawn_mode or bodo.tests.utils.test_spawn_mode_enabled:
        spawner = bodo.spawn.spawner.get_spawner()
        spawner.register_type(type_name, type_value)


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
        and B.key_arr_type.dtype == bodo.types.string_type
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

    # Shortcut for the common case
    if val_typ == typ:
        return val

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
            f"Invalid Python output data type specified.\nUser specified:\t{typ}\nValue type:\t{val_typ}"
        )

    print(val)
    return val


def bodosql_case_placeholder(arrs, case_code, out_arr_type):
    pass


@infer_global(bodosql_case_placeholder)
class CasePlaceholderTyper(AbstractTemplate):
    """Typing for BodoSQL CASE placeholder that will be replaced in dataframe pass."""

    def generic(self, args, kws):
        # last argument is the output array type provided by BodoSQL
        output_type = unwrap_typeref(args[-1])
        # Typing pass handles unknown output type for this
        assert_bodo_error(output_type != types.unknown)
        return signature(output_type, *args)


CasePlaceholderTyper.prefer_literal = True


def handle_bodosql_case_init_code(init_code):
    """Extract variable names in CASE initialization code generated by BodoSQL and check
    if we can avoid inlining.

    Args:
        init_code (str): CASE initialization code generated by BodoSQL

    Returns:
        tuple(list(str), bool): variable names and must_inline flag
    """
    # Extract the arrays used in the codegen to verify we can avoid inlining.
    var_names = []
    init_lines = init_code.split("\n")
    must_inline = False
    for line in init_lines:
        parts = line.split("=")
        # If len parts is 1, then this isn't an assignment
        # Right now everything is required to be an assignment
        # except empty whitespace at the BodoSQL level or closures.
        # We don't yet know how to avoid inlining the IR if we have
        # closures
        if len(parts) > 1:
            var_names.append(parts[0].strip())
        elif parts[0].strip() != "":
            must_inline = True
            break

    return var_names, must_inline


def gen_bodosql_case_func(
    init_code,
    body_code,
    named_param_args,
    var_names,
    arr_variable_name,
    indexing_variable_name,
    out_arr_type,
    func_globals,
    skip_allocation=False,
):
    """Generate a function for BodoSQL CASE statement using provided initialization
     code, loop body code, etc.

    Args:
        init_code (str): initialization code
        body_code (str): loop body code
        named_param_args (str): named parameter arguments
        var_names (list(str)): names of variables created in variable  initialization code
        arr_variable_name (str): output array's variable name
        indexing_variable_name (str): loop index variable name
        out_arr_type (types.Type): output type inferred by BodoSQL
        func_globals (dict): main function's globals
        skip_allocation (bool, optional): flag to skip output array allocation, used in typing pass to infer the scalar type. Defaults to False.

    Returns:
        function: generated function
    """
    import re

    import bodosql

    # TODO: Support named params
    func_text = f"def f(arrs, n, {named_param_args}):\n"
    func_text += init_code
    call_args = ", ".join(var_names)
    func_text += f"  return bodosql_case_kernel(({call_args},))\n"

    inner_func_text = "def bodosql_case_kernel(arrs):\n"
    for i, varname in enumerate(var_names):
        # Reuse the same variable name as the original query
        inner_func_text += f"  {varname} = arrs[{i}]\n"
    # Derive the length from the input array
    inner_func_text += f"  n = len({var_names[0]})\n"
    if not skip_allocation:
        inner_func_text += f"  {arr_variable_name} = bodo.utils.utils.alloc_type(n, out_arr_type, (-1,))\n"
    inner_func_text += f"  for {indexing_variable_name} in range(n):\n"
    inner_func_text += body_code
    inner_func_text += f"  return {arr_variable_name}\n"

    loc_vars = {}
    inner_glbls = {
        "pd": pd,
        "np": np,
        "re": re,
        "bodo": bodo,
        "bodosql": bodosql,
        "out_arr_type": out_arr_type,
    }
    # Globals generated by BodoSQL (accessible from main function) may be
    # necessary too.
    # See https://bodo.atlassian.net/browse/BSE-1941
    inner_glbls.update(func_globals)
    exec(inner_func_text, inner_glbls, loc_vars)
    inner_func = loc_vars["bodosql_case_kernel"]
    inner_jit = bodo.jit(distributed=False)(inner_func)
    glbls = {
        "numba": numba,
        "pd": pd,
        "np": np,
        "re": re,
        "bodo": bodo,
        "bodosql": bodosql,
        "bodosql_case_kernel": inner_jit,
    }
    # Globals generated by BodoSQL (accessible from main function) may be
    # necessary too.
    # See https://bodo.atlassian.net/browse/BSE-1941
    glbls.update(func_globals)
    exec(func_text, glbls, loc_vars)
    f = loc_vars["f"]
    return f, glbls


# NOTE: not using gen_objmode_func_overload since inspect cannot find the function
# signature for warnings.warn as of Python 3.12
if PYVERSION >= (3, 12):

    @overload(warnings.warn)
    def overload_warn(
        message, category=None, stacklevel=1, source=None, skip_file_prefixes=None
    ):
        def overload_warn_impl(
            message, category=None, stacklevel=1, source=None, skip_file_prefixes=None
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                with bodo.ir.object_mode.no_warning_objmode:
                    if skip_file_prefixes is None:
                        skip_file_prefixes = ()
                    warnings.warn(
                        message,
                        category,
                        stacklevel,
                        source,
                        skip_file_prefixes=skip_file_prefixes,
                    )

        return overload_warn_impl

else:
    gen_objmode_func_overload(warnings.warn, "none")


def get_array_getitem_scalar_type(t):
    """Returns scalar type of the array as returned by its getitem.

    Args:
        t (types.Type): input array type

    Returns:
        types.Type: scalar type (e.g int64, Timestamp, etc)
    """
    # Scalar type of most arrays is the same as dtype (e.g. int64), except
    # DatetimeArrayType and null_array_type which have different dtype objects.
    if isinstance(t, bodo.types.DatetimeArrayType):
        return bodo.types.PandasTimestampType(t.tz)

    if t == bodo.types.null_array_type:
        return types.none

    return t.dtype


def get_castable_arr_dtype(arr_type: types.Type):
    """Convert a Bodo array type into a Type representation
    that can be used for casting an array via fix_arr_dtype.

    Args:
        arr_type (types.Type): The array type to convert to a castable value.

    Returns:
        Any: The value used to generate the cast value.
    """
    if isinstance(
        arr_type,
        (
            bodo.types.ArrayItemArrayType,
            bodo.types.MapArrayType,
            bodo.types.StructArrayType,
        ),
    ):
        cast_typ = arr_type
    elif isinstance(
        arr_type, (bodo.types.IntegerArrayType, bodo.types.FloatingArrayType)
    ):
        cast_typ = arr_type.get_pandas_scalar_type_instance.name
    elif arr_type == bodo.types.boolean_array_type:
        cast_typ = bodo.libs.bool_arr_ext.boolean_dtype
    elif arr_type == bodo.types.dict_str_arr_type or isinstance(
        arr_type, bodo.types.DatetimeArrayType
    ):
        cast_typ = arr_type
    else:
        # Most array types cast using the dtype.
        cast_typ = arr_type.dtype
    return cast_typ


def is_bodosql_integer_arr_type(arr_typ: types.ArrayCompatible) -> bool:
    """Returns if a given array type may represent
    an integer type in BodoSQL. This is used when
    casting is required.

    Args:
        arr_typ (types.ArrayCompatible): The type to check.

    Returns:
        bool: Is the array a decimal or integer array (nullable or non-nullable).
    """
    return isinstance(arr_typ, bodo.types.DecimalArrayType) or isinstance(
        arr_typ.dtype, types.Integer
    )


def get_common_bodosql_integer_arr_type(
    arr_typs: list[types.ArrayCompatible],
) -> types.ArrayCompatible:
    """Returns a common array type for the BodoSQL integer array representations
    that have already been validated by is_bodosql_integer_type.

    Args:
        arr_typs (list[types.ArrayCompatible]): A list of integer (nullable or non-nullable) or decimal array to unify.

    Returns:
        types.ArrayCompatible: The output array with max bidwidth + correct nullability.
    """
    # Make sure arrays have the same nullability.
    to_nullable = any(is_nullable(arr_typ) for arr_typ in arr_typs)
    bitwidths = [arr_typ.dtype.bitwidth for arr_typ in arr_typs]
    max_bitwidth = max(bitwidths)
    typ_idx = bitwidths.index(max_bitwidth)
    arr_typ = arr_typs[typ_idx]
    if to_nullable:
        arr_typ = to_nullable_type(arr_typ)
    return arr_typ


def error_on_unsupported_streaming_arrays(table_type):
    """Raises an error if input table type has unsupported (Interval) nested arrays

    Args:
        table_type (TableType|unknown): input table type or unknown

    Raises:
        BodoError: error on nested arrays in input table type
    """
    # ignore unresolved types in typing pass
    if table_type in (None, types.unknown, types.undefined):
        return

    assert isinstance(table_type, bodo.types.TableType), (
        "error_on_unsupported_streaming_arrays: TableType expected"
    )

    for arr_type in table_type.arr_types:
        if isinstance(arr_type, bodo.types.IntervalArrayType):
            raise BodoError(f"Array type {arr_type} not supported in streaming yet")


class ExternalFunctionErrorChecked(types.ExternalFunction):
    """Same as Numba's ExternalFunction, but lowering checks for Python exceptions"""

    pass


register_model(ExternalFunctionErrorChecked)(models.OpaqueModel)
