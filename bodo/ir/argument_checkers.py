from abc import ABCMeta, abstractmethod
from types import NoneType

from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    is_literal_type,
)


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

        values_str = format_requirements_list(self._get_values_str, self.values, False)
        raise BodoError(
            f"{path}: Expected {self.arg_name} to be a compile time constant and must be {values_str}. Got: {arg_type}"
        )

    def explain_arg(self, context):
        values_str = format_requirements_list(self._get_values_str, self.values, True)
        return f"must be a compile time constant and must be {values_str}"


class ConstantArgumentChecker(AbstractArgumentTypeChecker):
    def __init__(self, arg_name, types):
        self.arg_name = arg_name
        self.types = tuple(types)

        self.types_to_str = {
            int: "integer",
            str: "string",
            bool: "boolean",
            NoneType: "none",
            tuple: "tuple",
            dict: "dict",
            list: "list",
        }

    def _get_types_str(self, typ):
        return self.types_to_str[typ] if typ in self.types_to_str else str(typ)

    def check_arg(self, context, path, arg_type):
        if is_literal_type(arg_type):
            val = get_literal_value(arg_type)
            if isinstance(val, self.types):
                return val

        types_str = format_requirements_list(
            self._get_types_str, self.types, usetick=False
        )
        raise BodoError(
            f"{path}: Expected {self.arg_name} to be a compile time constant and must have type {types_str}. Got: {arg_type}."
        )

    def explain_arg(self, context):
        types_str = format_requirements_list(
            self._get_types_str, self.types, usetick=True
        )
        return f"must be a compile time constant and must have type {types_str}"


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
