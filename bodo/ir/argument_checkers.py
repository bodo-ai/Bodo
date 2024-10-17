from abc import ABCMeta, abstractmethod
from types import NoneType

from numba.core import types

from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    is_iterable_type,
    is_literal_type,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_float,
    is_overload_int,
    is_overload_none,
    is_overload_numeric_scalar,
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

    def explain_arg(self, context):
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

    def explain_arg(self, context):
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
                f"{path}: Expected '{self.arg_name}' to be type {self.type_name}. Got: {arg_type}."
            )
        return arg_type

    def explain_arg(self, context):
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
        is_overload_str = lambda t: isinstance(
            t, types.UnicodeType
        ) or is_overload_constant_str(t)
        super(StringScalarArgumentChecker, self).__init__(
            arg_name, "String", is_overload_str
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
                f"{path}: Expected '{self.arg_name}' to be type {types_str}. Got: {arg_type}."
            )
        return arg_type

    def explain_arg(self, context):
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

    def explain_arg(self, context):
        return "must be a numeric scalar or Series, Index, Array, List, or Tuple with numeric data"


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
        """Creates a dictioanry mapping argument names to their description"""
        return {
            arg_name: arg_checker.explain_arg(self.context)
            for arg_name, arg_checker in self.argument_checkers.items()
        }
