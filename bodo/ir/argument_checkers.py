from abc import ABCMeta, abstractmethod
from types import NoneType

from numba.core import types

from bodo.hiframes.pd_series_ext import (
    SeriesType,
    is_dt64_series_typ,
    is_timedelta64_series_typ,
)
from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    get_overload_const_str_len,
    is_iterable_type,
    is_literal_type,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_float,
    is_overload_int,
    is_overload_none,
    is_overload_numeric_scalar,
    is_overload_str,
    is_str_arr_type,
)

_types_to_str = {
    int: "Integer",
    str: "String",
    bool: "Boolean",
    NoneType: "None",
    tuple: "Tuple",
    dict: "Dict",
    list: "List",
}


def format_requirements_list(to_string_elem, elems, usetick):
    """Format a list of requirements **elems** as a comma separated list where
    the last element is separated by an "or"

    Args:
        to_string_elem: Function mapping requirements in elems to an equivalent string representation
        elems: The list of requirements
        usetick (Boolean): Whether to wrap requirements with ` (for documentation-style formatting)

    Returns (str): The list of requirements
    """

    def to_string(elem):
        tick = "`" if usetick else ""
        elem_as_str = to_string_elem(elem)
        return f"{tick}{elem_as_str}{tick}"

    if len(elems) == 1:
        return to_string(elems[0])

    elems_as_strs = [to_string(elem) for elem in elems]

    return ", ".join(elems_as_strs[:-1]) + " or " + elems_as_strs[-1]


class AbstractArgumentTypeChecker(metaclass=ABCMeta):
    @abstractmethod
    def check_arg(self, context, path, arg_type):
        """Verify that **arg_type** is valid given the **context**"""

    @abstractmethod
    def explain_arg(self, context):
        """Generates a docstring for the given some **context**"""


class NDistinctValueArgumentChecker(AbstractArgumentTypeChecker):
    def __init__(self, arg_name, values):
        self.arg_name = arg_name
        self.values = values

    def _get_values_str(self, val):
        return f'"{val}"' if isinstance(val, str) else str(val)

    def check_arg(self, context, path, arg_type):
        if is_literal_type(arg_type):
            val = get_literal_value(arg_type)
            if val in self.values:
                return val
        elif arg_type in self.values:
            # check default argument case
            return arg_type

        values_str = format_requirements_list(self._get_values_str, self.values, False)
        raise BodoError(
            f"{path}: Expected '{self.arg_name}' to be a compile time constant and must be {values_str}. Got: {arg_type}."
        )

    def explain_arg(self, context):  # pragma: no cover
        values_str = format_requirements_list(self._get_values_str, self.values, True)
        return f"must be a compile time constant and must be {values_str}"


class ConstantArgumentChecker(AbstractArgumentTypeChecker):
    def __init__(self, arg_name, types):
        self.arg_name = arg_name
        self.types = tuple(types)

    def _get_types_str(self, typ):
        return _types_to_str[typ] if typ in _types_to_str else str(typ)

    def check_arg(self, context, path, arg_type):
        if is_literal_type(arg_type):
            val = get_literal_value(arg_type)
            if isinstance(val, self.types):
                return val
        elif isinstance(arg_type, self.types):
            # check default argument case
            return arg_type

        types_str = format_requirements_list(
            self._get_types_str, self.types, usetick=False
        )
        raise BodoError(
            f"{path}: Expected '{self.arg_name}' to be a constant {types_str}. Got: {arg_type}."
        )

    def explain_arg(self, context):  # pragma: no cover
        types_str = format_requirements_list(
            self._get_types_str, self.types, usetick=True
        )
        return f"must be a compile time constant and must be type {types_str}"


class PrimitiveTypeArgumentChecker(AbstractArgumentTypeChecker):
    def __init__(self, arg_name, type_name, is_overload_typ):
        self.arg_name = arg_name
        self.type_name = type_name
        self.is_overload_typ = is_overload_typ

    def check_arg(self, context, path, arg_type):
        if not self.is_overload_typ(arg_type):
            raise BodoError(
                f"{path}: Expected '{self.arg_name}' to be a {self.type_name}. Got: {arg_type}."
            )
        return arg_type

    def explain_arg(self, context):  # pragma: no cover
        return f"must be type `{self.type_name}`"


class IntegerScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name):
        super(IntegerScalarArgumentChecker, self).__init__(
            arg_name, "Integer", is_overload_int
        )


class BooleanScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name):
        super(BooleanScalarArgumentChecker, self).__init__(
            arg_name, "Boolean", is_overload_bool
        )


class FloatScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name):
        super(FloatScalarArgumentChecker, self).__init__(
            arg_name, "Float", is_overload_float
        )


class StringScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name):
        super(StringScalarArgumentChecker, self).__init__(
            arg_name, "String", is_overload_str
        )


class CharScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    """Checks that a value is a String and checks length is 1 if it can be determined at compile time"""

    def __init__(self, arg_name):
        is_overload_const_char_or_str = lambda t: (
            isinstance(t, types.UnicodeType)
        ) or (is_overload_constant_str(t) and get_overload_const_str_len(t) == 1)
        super(CharScalarArgumentChecker, self).__init__(
            arg_name, "Character", is_overload_const_char_or_str
        )


class NumericScalarArgumentChecker(AbstractArgumentTypeChecker):
    """
    Checker for arguments that can either be float or integer or None
    """

    def __init__(self, arg_name, is_optional=True):
        self.arg_name = arg_name
        self.is_optional = is_optional

    def check_arg(self, context, path, arg_type):
        if not (
            (self.is_optional and is_overload_none(arg_type))
            or is_overload_numeric_scalar(arg_type)
        ):
            types_str = (
                "Integer, Float, Boolean or None"
                if self.is_optional
                else "Integer, Float or Boolean"
            )
            raise BodoError(
                f"{path}: Expected '{self.arg_name}' to be a {types_str}. Got: {arg_type}."
            )
        return arg_type

    def explain_arg(self, context):  # pragma: no cover
        return (
            "must be `Integer`, `Float`, `Boolean`, or `None`"
            if self.is_optional
            else "must be `Integer`, `Float` or `Boolean"
        )


class NumericSeriesBinOpChecker(AbstractArgumentTypeChecker):
    """
    Checker for arguments that can be float or integer scalar or iterable with 1-d numeric data such as
    list, tuple, Series, Index, etc. Intended for for Series Binop methods such as Series.sub
    """

    def __init__(self, arg_name):
        self.arg_name = arg_name

    def check_arg(self, context, path, arg_typ):
        """Can either be numeric Scalar, or iterable with numeric data"""
        is_numeric_scalar = is_overload_numeric_scalar(arg_typ)
        is_numeric_iterable = is_iterable_type(arg_typ) and (
            isinstance(arg_typ.dtype, types.Number) or arg_typ.dtype == types.bool_
        )
        if not (is_numeric_scalar or is_numeric_iterable):
            raise BodoError(
                f"{path}: Expected '{self.arg_name}' to be a numeric scalar or Series, Index, Array, List or Tuple with numeric data: Got: {arg_typ}."
            )
        return arg_typ

    def explain_arg(self, context):  # pragma: no cover
        return "must be a numeric scalar or Series, Index, Array, List, or Tuple with numeric data"


class AnySeriesArgumentChecker(AbstractArgumentTypeChecker):
    """
    Argument checker for explicitly stating/documenting Series with any data are supported
    """

    def __init__(self, arg_name, is_self=False):
        self.arg_name = arg_name
        self.display_arg_name = "self" if is_self else self.arg_name
        self.is_self = is_self

    def check_arg(self, context, path, arg_type):
        if not isinstance(arg_type, SeriesType):
            raise BodoError(f"{path}: Expected {self.arg_name} to be a Series. Got:.")
        return arg_type

    def explain_arg(self, context):  # pragma: no cover
        return "all Series types supported"


class DatetimeLikeSeriesArgumentChecker(AnySeriesArgumentChecker):
    """
    Checker for documenting methods/attributes found in Series.dt
    """

    def __init__(self, arg_name, is_self=False, type="any"):
        super(DatetimeLikeSeriesArgumentChecker, self).__init__(arg_name, is_self)
        self.type = type

        # any: datetime or timedelta types accepted
        assert self.type in ["any", "datetime", "timedelta"]

    def check_arg(self, context, path, arg_type):
        """Check that arg_type is a Series of valid datetimelike data"""
        # Access underlying Series type for XMethodType (using getattr to avoid the circular import)
        series_type = getattr(arg_type, "stype", arg_type)

        if (
            self.type in ["any", "timedelta"] and is_timedelta64_series_typ(series_type)
        ) or (self.type in ["any", "datetime"] and is_dt64_series_typ(series_type)):
            return series_type

        if self.type == "any":
            supported_types = "datetime64 or timedelta64"
        else:
            supported_types = f"{self.type}64"

        raise BodoError(
            f"{path}: Expected '{self.display_arg_name}' to be a Series of {supported_types} data. Got: {series_type}"
        )

    def explain_arg(self, context):  # pragma: no cover
        supported_types = (
            "`datetime64` or `timedelta64`"
            if self.type == "any"
            else f"`{self.type}64`"
        )
        return f"must be a Series of {supported_types} data"


class NumericSeriesArgumentChecker(AnySeriesArgumentChecker):
    """For Series Arguments that require numeric data (Float or Integer)"""

    def check_arg(self, context, path, arg_type):
        if not isinstance(arg_type, SeriesType) or not isinstance(
            arg_type.dtype, types.Number
        ):
            raise BodoError(
                f"{path}: Expected '{self.display_arg_name}' to be a Series of Float or Integer data. Got: {arg_type}"
            )
        return arg_type

    def explain_arg(self, context):  # pragma: no cover
        return "must be a Series of `Integer` or `Float` data"


class StringSeriesArgumentChecker(AnySeriesArgumentChecker):
    """For Series arguments that require String data"""

    def check_arg(self, context, path, arg_type):
        """Check that the underlying data of Seires is a valid string type"""
        # Access underlying Series type for XMethodType (using getattr to avoid the circular import)
        series_type = getattr(arg_type, "stype", arg_type)
        if not (
            isinstance(series_type, SeriesType) and is_str_arr_type(series_type.data)
        ):
            raise BodoError(
                f"{path}: Expected '{self.display_arg_name}' to be a Series of String data. Got: {series_type}."
            )
        return series_type

    def explain_arg(self, context):  # pragma: no cover
        return "must be a Series of `String` data"


class AnyArgumentChecker(AbstractArgumentTypeChecker):
    """Dummy class for overload attribute that allows all types"""

    def __init__(self, arg_name):
        self.arg_name = arg_name

    def check_arg(self, context, path, arg_type):
        return arg_type

    def explain_arg(self, context):  # pragma: no cover
        return "supported on all datatypes"


class GenericArgumentChecker(AbstractArgumentTypeChecker):
    """
    Generic Argument type checker that accepts custom logic for check and explain.
    """

    def __init__(self, arg_name, check_fn, explain_fn, is_self=False):
        """
        Intialize an instance of GenericArgumentChecker with custom check and explain
        logic.

        Args:
            arg_name (str): The name of the argument.
            check_fn (Callable) a lambda which accepts a context containing all
                previously processed arguments and a type and returns a tuple where the
                first value indicates whether the argument is the correct type, if True
                then the second value contains the new argument type. If False, the
                second value is a string for error reporting.
            explain_str (Callable): a lambda that accepts a context containing all
                previously processed arguments and returns a string description of
                the typing rules for the argument.
            is_self (bool, optional): If true, display the argument name as "self" in
                error messages. Defaults to False.
        """
        self.arg_name = arg_name
        self.display_arg_name = "self" if is_self else self.arg_name
        self.check_fn = check_fn
        self.explain_fn = explain_fn

    def check_arg(self, context, path, arg_type):
        success, err_or_typ = self.check_fn(context, arg_type)
        if not success:
            raise BodoError(
                f"{path}: '{self.display_arg_name}' {err_or_typ}. Got: {arg_type}"
            )

        return err_or_typ

    def explain_arg(self, context):  # pragma: no cover
        return self.explain_fn(context)


class OverloadArgumentsChecker:
    def __init__(self, argument_checkers):
        self.argument_checkers = {
            arg_checker.arg_name: arg_checker for arg_checker in argument_checkers
        }
        self.context = {}

    def set_context(self, key, value):
        """Updates the type information of *key* in the Checker's internal context"""
        self.context.update({key: value})

    def check_args(self, path, arg_types):
        """Checks all argument listed in arg_types using argument_checkers"""
        for arg_name, typ in arg_types.items():
            if arg_name in self.argument_checkers:
                new_arg = self.argument_checkers[arg_name].check_arg(
                    self.context, path, typ
                )
                self.set_context(arg_name, new_arg)

    def explain_args(self):
        """Creates a dictionary mapping argument names to their description"""
        return {
            arg_name: arg_checker.explain_arg(self.context)
            for arg_name, arg_checker in self.argument_checkers.items()
        }


class OverloadAttributeChecker(OverloadArgumentsChecker):
    """Checker for attributes that accepts a single ArgumentChecker"""

    def __init__(self, argument_checker):
        self.argument_checker = argument_checker
        self.context = {}

    def check_args(self, path, arg_type):
        new_arg_type = self.argument_checker.check_arg(self.context, path, arg_type)
        self.set_context(self.argument_checker.arg_name, new_arg_type)

    def explain_args(self):
        return self.argument_checker.explain_arg(self.context)
